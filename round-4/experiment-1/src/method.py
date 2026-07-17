#!/usr/bin/env python3
"""OLS Slopes, Mid-SC Baseline, Benign Overlap, and |cos| Reconciliation.

Extends iter3 SAE Reversal Score analysis with four new measurements:
1. OLS no-intercept slope β_f per top-500 feature (magnitude vs direction)
2. Mid-SC baseline (40th–60th percentile) for noise-free null comparison
3. Benign-SC overlap to confirm harm-specificity of top-500 selection
4. |cos| numerical reconciliation across layers and iterations
"""

import gc
import json
import math
import os
import resource
import sys
from pathlib import Path

import numpy as np
import psutil
import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from scipy import stats

# ============================================================
# LOGGING
# ============================================================
WORKSPACE = Path("/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_4/gen_art/gen_art_experiment_1")
LOGS_DIR = WORKSPACE / "logs"
LOGS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = WORKSPACE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ============================================================
# HARDWARE & MEMORY SETUP
# ============================================================
def _container_ram_gb() -> float:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return psutil.virtual_memory().total / 1e9

TOTAL_RAM_GB = _container_ram_gb()
logger.info(f"Container RAM: {TOTAL_RAM_GB:.1f} GB")

# Set memory limit to 22 GB (leave buffer for OS and agent)
RAM_BUDGET = int(22 * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"RAM budget set to 22 GB")

# ============================================================
# PATHS
# ============================================================
os.environ.setdefault("HF_HOME", "/ai-inventor/aii_data/hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/ai-inventor/aii_data/hf_cache/hub")

ITER3_ACTS = Path("/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_3/gen_art/gen_art_experiment_1/activations")
DATA_PATH = Path("/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/results/data_out.json")
SAE_REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def pearson_batch(z_base: np.ndarray, z_inst: np.ndarray, z_abl: np.ndarray,
                  feat_idx: np.ndarray, prompt_idx: np.ndarray) -> np.ndarray:
    """Vectorized Pearson R for each feature in feat_idx. Returns (F,) float32."""
    di = z_inst[np.ix_(prompt_idx, feat_idx)] - z_base[np.ix_(prompt_idx, feat_idx)]
    da = z_abl[np.ix_(prompt_idx, feat_idx)] - z_base[np.ix_(prompt_idx, feat_idx)]
    di_c = di - di.mean(axis=0, keepdims=True)
    da_c = da - da.mean(axis=0, keepdims=True)
    num = (di_c * da_c).sum(axis=0)
    denom = np.sqrt((di_c**2).sum(axis=0) * (da_c**2).sum(axis=0)) + 1e-12
    return (num / denom).astype(np.float32)


def load_sae(layer_idx: int):
    """Load SAE checkpoint; return W_enc (65536,4096), b_enc (65536,), W_dec (4096,65536)."""
    logger.info(f"Loading SAE checkpoint for layer {layer_idx}...")
    ckpt_path = hf_hub_download(repo_id=SAE_REPO, filename=f"layer{layer_idx}.sae.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    logger.debug(f"  SAE keys: {list(ckpt.keys())}")
    W_enc = ckpt["W_enc"].numpy().astype(np.float32)  # (65536, 4096)
    b_enc = ckpt["b_enc"].numpy().astype(np.float32)  # (65536,)
    W_dec = ckpt["W_dec"].numpy().astype(np.float32)  # (4096, 65536)
    assert W_enc.shape == (65536, 4096), f"Unexpected W_enc shape: {W_enc.shape}"
    logger.info(f"  SAE L{layer_idx}: W_enc {W_enc.shape}, b_enc {b_enc.shape}, W_dec {W_dec.shape}")
    return W_enc, b_enc, W_dec


def compute_pre_topk(act_arr: np.ndarray, W_enc: np.ndarray, b_enc: np.ndarray, chunk: int = 200) -> np.ndarray:
    """Pre-topk projections: z = act @ W_enc.T + b_enc. Returns (N, 65536) float32."""
    N = act_arr.shape[0]
    out = np.empty((N, 65536), dtype=np.float32)
    for i in range(0, N, chunk):
        out[i:i + chunk] = act_arr[i:i + chunk] @ W_enc.T + b_enc
    return out


def cos_to_r(r: np.ndarray, W_enc: np.ndarray, feat_idx: np.ndarray) -> np.ndarray:
    """Return signed cos(r, W_enc[f]) for each f in feat_idx."""
    vecs = W_enc[feat_idx]  # (F, 4096)
    norms = np.linalg.norm(vecs, axis=1) + 1e-12
    return (vecs @ r) / norms  # (F,)


def ols_slopes(z_base: np.ndarray, z_inst: np.ndarray, z_abl: np.ndarray,
               feat_idx: np.ndarray, prompt_idx: np.ndarray) -> np.ndarray:
    """OLS no-intercept slope β_f = dot(δ_inst_f, δ_abl_f) / dot(δ_inst_f, δ_inst_f)."""
    di = z_inst[np.ix_(prompt_idx, feat_idx)] - z_base[np.ix_(prompt_idx, feat_idx)]
    da = z_abl[np.ix_(prompt_idx, feat_idx)] - z_base[np.ix_(prompt_idx, feat_idx)]
    numer = (di * da).sum(axis=0)
    denom = (di * di).sum(axis=0)
    # Exclude near-zero denominator features
    valid = denom > 0.01
    beta = np.full(len(feat_idx), np.nan, dtype=np.float32)
    beta[valid] = numer[valid] / denom[valid]
    n_excluded = int((~valid).sum())
    if n_excluded > 0:
        logger.warning(f"  OLS: excluded {n_excluded}/{len(feat_idx)} features with near-zero variance")
    return beta


@logger.catch(reraise=True)
def main():
    logger.info("=" * 60)
    logger.info("iter4 experiment: OLS slopes, mid-SC baseline, benign overlap, |cos| reconciliation")
    logger.info("=" * 60)

    # ============================================================
    # STEP 0 — LOAD DATA AND CACHED ACTIVATIONS
    # ============================================================
    logger.info("Loading dataset...")
    data = json.loads(DATA_PATH.read_text())
    logger.info(f"  Loaded {len(data)} prompts")

    harmful_idx = np.array([i for i, d in enumerate(data) if d["label"] == "harmful"])
    benign_idx = np.array([i for i, d in enumerate(data) if d["label"] == "benign"])
    assert len(harmful_idx) == 500, f"Expected 500 harmful, got {len(harmful_idx)}"
    assert len(benign_idx) == 500, f"Expected 500 benign, got {len(benign_idx)}"
    logger.info(f"  harmful_idx: {len(harmful_idx)}, benign_idx: {len(benign_idx)}")

    logger.info("Loading cached activations from iter3...")
    act_files = {
        "base_L22": ITER3_ACTS / "base_act_L22.npy",
        "inst_L22": ITER3_ACTS / "inst_act_L22.npy",
        "abl_L22": ITER3_ACTS / "abl_act_L22.npy",
        "base_L20": ITER3_ACTS / "base_act_L20.npy",
        "inst_L20": ITER3_ACTS / "inst_act_L20.npy",
        "abl_L20": ITER3_ACTS / "abl_act_L20.npy",
    }
    for name, path in act_files.items():
        assert path.exists(), f"Missing activation file: {path}"

    base_L22 = np.load(act_files["base_L22"]).astype(np.float32)
    inst_L22 = np.load(act_files["inst_L22"]).astype(np.float32)
    abl_L22 = np.load(act_files["abl_L22"]).astype(np.float32)
    base_L20 = np.load(act_files["base_L20"]).astype(np.float32)
    inst_L20 = np.load(act_files["inst_L20"]).astype(np.float32)
    abl_L20 = np.load(act_files["abl_L20"]).astype(np.float32)

    for name, arr in [("base_L22", base_L22), ("inst_L22", inst_L22), ("abl_L22", abl_L22),
                       ("base_L20", base_L20), ("inst_L20", inst_L20), ("abl_L20", abl_L20)]:
        assert np.isfinite(arr).all(), f"NaN/Inf in {name}"
        assert arr.shape == (1000, 4096), f"Unexpected shape in {name}: {arr.shape}"
        logger.info(f"  {name}: shape={arr.shape}, mean={arr.mean():.4f}, std={arr.std():.4f}")

    # ============================================================
    # STEP 1 — SAE L22: PRE-TOPK PROJECTIONS
    # ============================================================
    logger.info("Loading SAE L22 and computing pre-topk projections...")
    W_enc22, b_enc22, W_dec22 = load_sae(22)

    logger.info("  Computing z_base22...")
    z_base22 = compute_pre_topk(base_L22, W_enc22, b_enc22)
    logger.info("  Computing z_inst22...")
    z_inst22 = compute_pre_topk(inst_L22, W_enc22, b_enc22)
    logger.info("  Computing z_abl22...")
    z_abl22 = compute_pre_topk(abl_L22, W_enc22, b_enc22)

    for name, arr in [("z_base22", z_base22), ("z_inst22", z_inst22), ("z_abl22", z_abl22)]:
        assert np.isfinite(arr).all(), f"NaN/Inf in {name}"
        logger.info(f"  {name}: mean={arr.mean():.4f}, std={arr.std():.4f}")

    # ============================================================
    # STEP 2 — FEATURE INDEX SETS
    # ============================================================
    logger.info("Computing Safety Change (SC) on harmful prompts at L22...")
    SC_harm22 = np.mean(np.abs(z_inst22[harmful_idx] - z_base22[harmful_idx]), axis=0)
    SC_benign22 = np.mean(np.abs(z_inst22[benign_idx] - z_base22[benign_idx]), axis=0)

    top500_idx = np.argsort(SC_harm22)[-500:][::-1]
    bottom500_idx = np.argsort(SC_harm22)[:500]
    assert len(np.intersect1d(top500_idx, bottom500_idx)) == 0, "top/bottom overlap!"

    logger.info(f"  top500: min SC={SC_harm22[top500_idx].min():.4f}, max SC={SC_harm22[top500_idx].max():.4f}")
    logger.info(f"  bottom500: max SC={SC_harm22[bottom500_idx].max():.4f}")

    # Cross-check with iter3 known values
    top500_sc_min = float(SC_harm22[top500_idx].min())
    bottom500_sc_max = float(SC_harm22[bottom500_idx].max())
    if abs(top500_sc_min - 2.787) > 0.05:
        logger.warning(f"  top500_sc_min={top500_sc_min:.4f} differs from iter3=2.787 by >{0.05}")
    if abs(bottom500_sc_max - 0.682) > 0.05:
        logger.warning(f"  bottom500_sc_max={bottom500_sc_max:.4f} differs from iter3=0.682 by >{0.05}")

    # MID-SC: 40th–60th percentile
    p40 = float(np.percentile(SC_harm22, 40))
    p60 = float(np.percentile(SC_harm22, 60))
    mid_candidates = np.where((SC_harm22 >= p40) & (SC_harm22 <= p60))[0]
    np.random.seed(42)
    if len(mid_candidates) >= 500:
        mid500_idx = np.random.choice(mid_candidates, size=500, replace=False)
    else:
        mid500_idx = mid_candidates
    logger.info(f"  Mid-SC range: [{p40:.4f}, {p60:.4f}], candidates={len(mid_candidates)}, selected={len(mid500_idx)}")
    logger.info(f"  Mid-SC median SC: {np.median(SC_harm22[mid500_idx]):.4f}")

    # BENIGN-SC top-500 and overlap
    top500_benign_idx = np.argsort(SC_benign22)[-500:]
    overlap_set = np.intersect1d(top500_idx, top500_benign_idx)
    benign_overlap_count = len(overlap_set)
    benign_overlap_fraction = benign_overlap_count / 500.0
    if benign_overlap_fraction < 0.20:
        benign_overlap_verdict = "HARM_SPECIFIC"
    elif benign_overlap_fraction < 0.60:
        benign_overlap_verdict = "MIXED_HARM_STYLE"
    else:
        benign_overlap_verdict = "STYLE_DOMINANT"
    logger.info(f"  Benign-SC overlap: {benign_overlap_count}/500 = {benign_overlap_fraction:.3f} → {benign_overlap_verdict}")

    # SC medians
    bottom500_SC_median = float(np.median(SC_harm22[bottom500_idx]))
    mid500_SC_median = float(np.median(SC_harm22[mid500_idx]))
    logger.info(f"  SC median — bottom500: {bottom500_SC_median:.4f}, mid500: {mid500_SC_median:.4f}")

    # ============================================================
    # STEP 3 — PEARSON R REVERSAL SCORES
    # ============================================================
    logger.info("Computing Pearson R reversal scores for all feature groups at L22...")
    R_top500 = pearson_batch(z_base22, z_inst22, z_abl22, top500_idx, harmful_idx)
    R_bottom500 = pearson_batch(z_base22, z_inst22, z_abl22, bottom500_idx, harmful_idx)
    R_mid500 = pearson_batch(z_base22, z_inst22, z_abl22, mid500_idx, harmful_idx)

    logger.info(f"  top500:    R_mean={R_top500.mean():.4f}, std={R_top500.std():.4f}")
    logger.info(f"  bottom500: R_mean={R_bottom500.mean():.4f}, std={R_bottom500.std():.4f}")
    logger.info(f"  mid500:    R_mean={R_mid500.mean():.4f}, std={R_mid500.std():.4f}")

    # Cross-check iter3 values
    if abs(R_top500.mean() - 0.9205) > 0.01:
        logger.warning(f"  R_top500_mean={R_top500.mean():.4f} differs from iter3=0.9205 by >0.01")
    else:
        logger.info(f"  Cross-check PASS: R_top500_mean={R_top500.mean():.4f} ≈ 0.9205")

    # Statistical tests
    ttest_top_mid = stats.ttest_ind(R_top500, R_mid500)
    kohens_d_top_mid = float((R_top500.mean() - R_mid500.mean()) /
                              np.sqrt((R_top500.var() + R_mid500.var()) / 2))
    ttest_top_bot = stats.ttest_ind(R_top500, R_bottom500)
    cohens_d_top_bot = float((R_top500.mean() - R_bottom500.mean()) /
                              np.sqrt((R_top500.var() + R_bottom500.var()) / 2))

    if ttest_top_mid.pvalue < 0.05 and kohens_d_top_mid > 0.5:
        mid_baseline_verdict = "MID_SC_CONFIRMS_SELECTIVE_PRESERVATION"
    elif ttest_top_mid.pvalue < 0.05:
        mid_baseline_verdict = "MID_SC_WEAK_SIGNIFICANCE"
    else:
        mid_baseline_verdict = "MID_SC_BASELINE_WEAKENS_CLAIM"
    logger.info(f"  top vs mid: t={ttest_top_mid.statistic:.3f}, p={ttest_top_mid.pvalue:.2e}, d={kohens_d_top_mid:.3f} → {mid_baseline_verdict}")
    logger.info(f"  top vs bot: t={ttest_top_bot.statistic:.3f}, p={ttest_top_bot.pvalue:.2e}, d={cohens_d_top_bot:.3f}")

    # ============================================================
    # STEP 4 — OLS NO-INTERCEPT SLOPE β_f FOR TOP-500 FEATURES (L22)
    # ============================================================
    logger.info("Computing OLS no-intercept slopes β_f for top-500 at L22...")
    beta_f = ols_slopes(z_base22, z_inst22, z_abl22, top500_idx, harmful_idx)

    # Report on valid (non-nan) betas
    valid_mask = np.isfinite(beta_f)
    beta_valid = beta_f[valid_mask]
    n_valid = int(valid_mask.sum())
    logger.info(f"  β_f (n={n_valid}): mean={beta_valid.mean():.4f}, std={beta_valid.std():.4f}, median={np.median(beta_valid):.4f}")
    logger.info(f"  β_f: min={beta_valid.min():.4f}, max={beta_valid.max():.4f}")
    logger.info(f"  β_f > 0.8: {(beta_valid > 0.8).mean():.3f} ({(beta_valid > 0.8).sum()}/{n_valid})")
    logger.info(f"  β_f > 1.0: {(beta_valid > 1.0).mean():.3f} ({(beta_valid > 1.0).sum()}/{n_valid})")
    logger.info(f"  β_f < 0.2: {(beta_valid < 0.2).mean():.3f} ({(beta_valid < 0.2).sum()}/{n_valid})")
    logger.info(f"  β_f < 0.0: {(beta_valid < 0.0).mean():.3f} ({(beta_valid < 0.0).sum()}/{n_valid})")

    beta_mean = float(beta_valid.mean())
    beta_std = float(beta_valid.std())
    if beta_mean > 0.8:
        magnitude_verdict = "MAGNITUDE_PRESERVED"
    elif beta_mean > 0.3:
        magnitude_verdict = "PARTIAL_ATTENUATION"
    else:
        magnitude_verdict = "DIRECTION_ONLY_ECHO"
    logger.info(f"  Magnitude verdict: {magnitude_verdict}")

    # Histogram (20 bins)
    beta_hist_counts, beta_hist_edges = np.histogram(beta_valid, bins=20)
    beta_hist = [{"lo": float(beta_hist_edges[i]), "hi": float(beta_hist_edges[i + 1]),
                  "count": int(beta_hist_counts[i])} for i in range(20)]

    # Also compute on all 1000 prompts (harmful + benign) as robustness check
    beta_f_all = ols_slopes(z_base22, z_inst22, z_abl22, top500_idx, np.arange(1000))
    beta_f_all_valid = beta_f_all[np.isfinite(beta_f_all)]
    logger.info(f"  β_f (all 1000 prompts): mean={beta_f_all_valid.mean():.4f}, std={beta_f_all_valid.std():.4f}")

    # Sanity check: β_f ≈ R_f × std(da_f)/std(di_f)
    # Spot check on 5 features
    logger.info("  Spot-checking OLS vs Pearson relationship for 5 features...")
    for k in range(min(5, 500)):
        f = top500_idx[k]
        di_k = (z_inst22[harmful_idx, f] - z_base22[harmful_idx, f])
        da_k = (z_abl22[harmful_idx, f] - z_base22[harmful_idx, f])
        r_k = float(stats.pearsonr(di_k, da_k)[0])
        std_ratio = float(da_k.std() / (di_k.std() + 1e-12))
        beta_expected = r_k * std_ratio
        logger.debug(f"    f={f}: R={r_k:.4f}, std_ratio={std_ratio:.4f}, beta_expected={beta_expected:.4f}, beta_actual={float(beta_f[k]):.4f}")

    # ============================================================
    # STEP 5 — |cos| RECONCILIATION AT L22
    # ============================================================
    logger.info("Computing refusal direction and |cos| at L22...")
    r22_raw = inst_L22[harmful_idx].mean(axis=0) - inst_L22[benign_idx].mean(axis=0)
    r22_norm = float(np.linalg.norm(r22_raw))
    r22 = r22_raw / (r22_norm + 1e-12)
    logger.info(f"  L22 refusal_dir_norm={r22_norm:.4f} (iter3: 37.53)")

    cos_enc_top500_L22 = cos_to_r(r22, W_enc22, top500_idx)
    cos_enc_bottom500_L22 = cos_to_r(r22, W_enc22, bottom500_idx)
    cos_enc_mid500_L22 = cos_to_r(r22, W_enc22, mid500_idx)

    cos_enc_top500_L22_mean_abs = float(np.abs(cos_enc_top500_L22).mean())
    cos_enc_bottom500_L22_mean_abs = float(np.abs(cos_enc_bottom500_L22).mean())
    cos_enc_mid500_L22_mean_abs = float(np.abs(cos_enc_mid500_L22).mean())
    logger.info(f"  L22 |cos_enc| top500: {cos_enc_top500_L22_mean_abs:.5f} (iter3: 0.01853)")
    logger.info(f"  L22 |cos_enc| bottom500: {cos_enc_bottom500_L22_mean_abs:.5f}")
    logger.info(f"  L22 |cos_enc| mid500: {cos_enc_mid500_L22_mean_abs:.5f}")
    if abs(cos_enc_top500_L22_mean_abs - 0.01853) > 0.005:
        logger.warning(f"  |cos| L22 differs from iter3 by >{0.005}: got {cos_enc_top500_L22_mean_abs:.5f}")
    else:
        logger.info(f"  Cross-check PASS: |cos| L22 ≈ 0.01853")

    # Free L22 SAE weights before loading L20
    del W_enc22, W_dec22
    gc.collect()
    logger.info("Freed L22 SAE weights.")

    # ============================================================
    # STEP 6 — SAE L20: PRE-TOPK PROJECTIONS AND METRICS
    # ============================================================
    logger.info("Loading SAE L20 and computing projections...")
    W_enc20, b_enc20, W_dec20 = load_sae(20)

    z_base20 = compute_pre_topk(base_L20, W_enc20, b_enc20)
    z_inst20 = compute_pre_topk(inst_L20, W_enc20, b_enc20)
    z_abl20 = compute_pre_topk(abl_L20, W_enc20, b_enc20)

    # SC at L20
    SC_harm20 = np.mean(np.abs(z_inst20[harmful_idx] - z_base20[harmful_idx]), axis=0)
    top500_L20_idx = np.argsort(SC_harm20)[-500:][::-1]
    bottom500_L20_idx = np.argsort(SC_harm20)[:500]

    # Refusal direction at L20
    r20_raw = inst_L20[harmful_idx].mean(axis=0) - inst_L20[benign_idx].mean(axis=0)
    r20_norm = float(np.linalg.norm(r20_raw))
    r20 = r20_raw / (r20_norm + 1e-12)
    logger.info(f"  L20 refusal_dir_norm={r20_norm:.4f}")

    cos_enc_top500_L20 = cos_to_r(r20, W_enc20, top500_L20_idx)
    cos_enc_bottom500_L20 = cos_to_r(r20, W_enc20, bottom500_L20_idx)
    cos_enc_top500_L20_mean_abs = float(np.abs(cos_enc_top500_L20).mean())
    logger.info(f"  L20 |cos_enc| top500: {cos_enc_top500_L20_mean_abs:.5f} (iter3: 0.02086)")

    # OLS slopes at L20
    beta_f_L20 = ols_slopes(z_base20, z_inst20, z_abl20, top500_L20_idx, harmful_idx)
    beta_f_L20_valid = beta_f_L20[np.isfinite(beta_f_L20)]
    logger.info(f"  L20 β_f: mean={beta_f_L20_valid.mean():.4f}, std={beta_f_L20_valid.std():.4f}")

    # Pearson R at L20
    R_top500_L20 = pearson_batch(z_base20, z_inst20, z_abl20, top500_L20_idx, harmful_idx)
    R_bottom500_L20 = pearson_batch(z_base20, z_inst20, z_abl20, bottom500_L20_idx, harmful_idx)
    logger.info(f"  L20 R_top500: {R_top500_L20.mean():.4f} (iter3: 0.9245)")
    if abs(R_top500_L20.mean() - 0.9245) > 0.01:
        logger.warning(f"  L20 R differs from iter3 by >0.01: got {R_top500_L20.mean():.4f}")
    else:
        logger.info(f"  Cross-check PASS: L20 R_top500_mean ≈ 0.9245")

    # Mid-SC at L20
    p40_L20 = float(np.percentile(SC_harm20, 40))
    p60_L20 = float(np.percentile(SC_harm20, 60))
    mid_candidates_L20 = np.where((SC_harm20 >= p40_L20) & (SC_harm20 <= p60_L20))[0]
    np.random.seed(43)
    if len(mid_candidates_L20) >= 500:
        mid500_L20_idx = np.random.choice(mid_candidates_L20, size=500, replace=False)
    else:
        mid500_L20_idx = mid_candidates_L20
    R_mid500_L20 = pearson_batch(z_base20, z_inst20, z_abl20, mid500_L20_idx, harmful_idx)
    logger.info(f"  L20 R_mid500: {R_mid500_L20.mean():.4f}")

    # Free L20 SAE weights
    del W_enc20, W_dec20
    gc.collect()
    logger.info("Freed L20 SAE weights.")

    # ============================================================
    # STEP 7 — COMPILE ANALYSIS SUMMARY
    # ============================================================
    logger.info("Compiling analysis summary...")

    analysis = {
        "meta": {
            "n_harmful": 500,
            "n_benign": 500,
            "n_features_total": 65536,
            "sae_repo": SAE_REPO,
            "activations_source": str(ITER3_ACTS),
            "primary_layer": 22,
            "replication_layer": 20,
        },
        "ols_slopes_L22": {
            "n_valid": n_valid,
            "n_excluded": int(500 - n_valid),
            "beta_mean": beta_mean,
            "beta_std": beta_std,
            "beta_median": float(np.median(beta_valid)),
            "beta_min": float(beta_valid.min()),
            "beta_max": float(beta_valid.max()),
            "fraction_gt_0p8": float((beta_valid > 0.8).mean()),
            "fraction_gt_1p0": float((beta_valid > 1.0).mean()),
            "fraction_lt_0p2": float((beta_valid < 0.2).mean()),
            "fraction_negative": float((beta_valid < 0.0).mean()),
            "histogram_20bins": beta_hist,
            "verdict": magnitude_verdict,
            "beta_f_all_prompts_mean": float(beta_f_all_valid.mean()),
            "beta_f_all_prompts_std": float(beta_f_all_valid.std()),
        },
        "ols_slopes_L20": {
            "n_valid": int(np.isfinite(beta_f_L20).sum()),
            "beta_mean": float(beta_f_L20_valid.mean()),
            "beta_std": float(beta_f_L20_valid.std()),
            "beta_median": float(np.median(beta_f_L20_valid)),
            "beta_min": float(beta_f_L20_valid.min()),
            "beta_max": float(beta_f_L20_valid.max()),
            "fraction_gt_0p8": float((beta_f_L20_valid > 0.8).mean()),
            "fraction_negative": float((beta_f_L20_valid < 0.0).mean()),
        },
        "mid_sc_baseline_L22": {
            "p40_sc": p40,
            "p60_sc": p60,
            "mid500_sc_median": mid500_SC_median,
            "bottom500_sc_median": bottom500_SC_median,
            "n_mid_candidates": int(len(mid_candidates)),
            "n_mid_selected": int(len(mid500_idx)),
            "mid500_R_mean": float(R_mid500.mean()),
            "mid500_R_std": float(R_mid500.std()),
            "mid500_R_min": float(R_mid500.min()),
            "mid500_R_max": float(R_mid500.max()),
            "top500_vs_mid500_ttest_t": float(ttest_top_mid.statistic),
            "top500_vs_mid500_ttest_p": float(ttest_top_mid.pvalue),
            "top500_vs_mid500_cohens_d": float(kohens_d_top_mid),
            "top500_vs_bottom500_cohens_d": float(cohens_d_top_bot),
            "verdict": mid_baseline_verdict,
            "noise_concern": "MITIGATED" if bottom500_SC_median > 0.01 else "VALID_CONCERN",
        },
        "mid_sc_baseline_L20": {
            "p40_sc": p40_L20,
            "p60_sc": p60_L20,
            "mid500_R_mean": float(R_mid500_L20.mean()),
            "mid500_R_std": float(R_mid500_L20.std()),
        },
        "benign_sc_overlap_L22": {
            "overlap_count": int(benign_overlap_count),
            "overlap_fraction": float(benign_overlap_fraction),
            "verdict": benign_overlap_verdict,
            "interpretation": (
                "<20% overlap → top-500 captures harm-specific RLHF changes, not style"
                if benign_overlap_fraction < 0.20
                else ">=20% overlap → RLHF changes in top-500 are partially style-driven"
            ),
        },
        "cos_reconciliation": {
            "iter4_L22_cos_enc_top500_mean_abs": cos_enc_top500_L22_mean_abs,
            "iter4_L20_cos_enc_top500_mean_abs": cos_enc_top500_L20_mean_abs,
            "iter4_L22_cos_enc_bottom500_mean_abs": cos_enc_bottom500_L22_mean_abs,
            "iter4_L22_cos_enc_mid500_mean_abs": cos_enc_mid500_L22_mean_abs,
            "iter4_L20_cos_enc_bottom500_mean_abs": float(np.abs(cos_enc_bottom500_L20).mean()),
            "iter3_reported_L22": 0.01853,
            "iter3_reported_L20": 0.02086,
            "paper_draft_L22": 0.019,
            "paper_draft_L20": 0.021,
            "hypothesis_summary_value": 0.0145,
            "discrepancy_explanation": (
                "iter3 L22=0.01853 ≈ paper 0.019 (rounding); L20=0.02086 ≈ paper 0.021 (rounding). "
                "The 0.0145 in hypothesis summary likely reflects a different top-500 set or "
                "an earlier iteration. iter4 recomputation is the authoritative value."
            ),
            "authoritative_cos_l22": cos_enc_top500_L22_mean_abs,
            "authoritative_cos_l20": cos_enc_top500_L20_mean_abs,
            "l22_refusal_dir_norm": r22_norm,
            "l20_refusal_dir_norm": r20_norm,
        },
        "null_baseline_crosscheck_L22": {
            "top500_R_mean": float(R_top500.mean()),
            "top500_R_std": float(R_top500.std()),
            "top500_R_min": float(R_top500.min()),
            "top500_R_max": float(R_top500.max()),
            "bottom500_R_mean": float(R_bottom500.mean()),
            "bottom500_R_std": float(R_bottom500.std()),
            "cohens_d_top_vs_bottom": float(cohens_d_top_bot),
            "ttest_top_vs_bottom_t": float(ttest_top_bot.statistic),
            "ttest_top_vs_bottom_p": float(ttest_top_bot.pvalue),
            "iter3_top500_R_mean": 0.9205,
            "iter3_bottom500_R_mean": 0.8470,
            "iter3_cohens_d": 1.7508,
        },
        "null_baseline_crosscheck_L20": {
            "top500_R_mean": float(R_top500_L20.mean()),
            "top500_R_std": float(R_top500_L20.std()),
            "bottom500_R_mean": float(R_bottom500_L20.mean()),
            "iter3_top500_R_mean": 0.9245,
        },
        "primary_verdict": magnitude_verdict,
        "paper_framing": (
            "comprehensive_latent_imprint" if magnitude_verdict == "MAGNITUDE_PRESERVED"
            else "directional_latent_imprint" if magnitude_verdict == "PARTIAL_ATTENUATION"
            else "geometric_echo"
        ),
    }

    analysis_path = RESULTS_DIR / "analysis_out.json"
    analysis_path.write_text(json.dumps(analysis, indent=2))
    logger.info(f"Saved analysis_out.json: {analysis_path}")

    # ============================================================
    # STEP 8 — BUILD exp_gen_sol_out (1000 rows)
    # ============================================================
    logger.info("Building per-prompt output (1000 rows)...")

    # Per-prompt mean |δ_inst| and |δ_abl| across top-500 features
    delta_inst_top500 = np.abs(z_inst22[:, top500_idx] - z_base22[:, top500_idx]).mean(axis=1)
    delta_abl_top500 = np.abs(z_abl22[:, top500_idx] - z_base22[:, top500_idx]).mean(axis=1)

    examples = []
    for i, d in enumerate(data):
        # Per-prompt OLS proxy
        di_i = z_inst22[i, top500_idx] - z_base22[i, top500_idx]
        da_i = z_abl22[i, top500_idx] - z_base22[i, top500_idx]
        local_beta_num = float((di_i * da_i).sum())
        local_beta_den = float((di_i * di_i).sum())
        local_beta_proxy = local_beta_num / (local_beta_den + 1e-20)

        delta_inst_i = float(delta_inst_top500[i])
        delta_abl_i = float(delta_abl_top500[i])
        delta_ratio = delta_abl_i / (delta_inst_i + 1e-10)

        examples.append({
            "input": d["prompt"],
            "output": d["label"],
            "predict_is_harmful": str(d["label"] == "harmful"),
            "predict_beta_proxy": f"{local_beta_proxy:.6f}",
            "predict_mean_delta_inst": f"{delta_inst_i:.6f}",
            "predict_mean_delta_abl": f"{delta_abl_i:.6f}",
            "predict_delta_ratio": f"{delta_ratio:.4f}",
            "predict_beta_mean_top500": f"{beta_mean:.4f}",
            "predict_r_mean_top500": f"{float(R_top500.mean()):.4f}",
            "predict_mid_sc_r_mean": f"{float(R_mid500.mean()):.4f}",
            "predict_benign_overlap": f"{benign_overlap_fraction:.4f}",
            "predict_cos_enc_l22_mean": f"{cos_enc_top500_L22_mean_abs:.5f}",
            "predict_magnitude_verdict": magnitude_verdict,
            "predict_mid_baseline_verdict": mid_baseline_verdict,
            "predict_benign_overlap_verdict": benign_overlap_verdict,
            "metadata_source": d.get("source", "unknown"),
            "metadata_category": d.get("category", "unknown"),
            "metadata_id": d.get("id", f"p_{i:04d}"),
            "metadata_char_len": d.get("char_len", len(d["prompt"])),
        })

    method_out = {
        "metadata": {
            "method_name": "OLS Slope and Mid-SC Baseline Analysis",
            "description": (
                "Extends iter3 Reversal Score analysis with four new measurements: "
                "(1) OLS no-intercept slope β_f for magnitude resolution, "
                "(2) mid-SC baseline for clean null comparison, "
                "(3) benign-SC overlap for harm-specificity, "
                "(4) |cos| numerical reconciliation."
            ),
            "beta_mean_top500_L22": analysis["ols_slopes_L22"]["beta_mean"],
            "beta_std_top500_L22": analysis["ols_slopes_L22"]["beta_std"],
            "magnitude_verdict": analysis["ols_slopes_L22"]["verdict"],
            "r_mean_top500_L22": float(R_top500.mean()),
            "mid_sc_r_mean_L22": analysis["mid_sc_baseline_L22"]["mid500_R_mean"],
            "mid_sc_baseline_verdict": analysis["mid_sc_baseline_L22"]["verdict"],
            "benign_overlap_fraction": analysis["benign_sc_overlap_L22"]["overlap_fraction"],
            "benign_overlap_verdict": analysis["benign_sc_overlap_L22"]["verdict"],
            "authoritative_cos_L22": analysis["cos_reconciliation"]["authoritative_cos_l22"],
            "authoritative_cos_L20": analysis["cos_reconciliation"]["authoritative_cos_l20"],
            "paper_framing": analysis["paper_framing"],
        },
        "datasets": [
            {
                "dataset": "harmful_benign_prompts_ols_analysis",
                "examples": examples,
            }
        ],
    }

    method_out_path = RESULTS_DIR / "method_out.json"
    method_out_path.write_text(json.dumps(method_out, indent=2))
    logger.info(f"Saved method_out.json: {method_out_path} ({len(examples)} examples)")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info(f"  PRIMARY VERDICT:      {magnitude_verdict}")
    logger.info(f"  β_f mean (L22):       {beta_mean:.4f} ± {beta_std:.4f}")
    logger.info(f"  R_top500 (L22):       {R_top500.mean():.4f} (iter3: 0.9205)")
    logger.info(f"  Mid-SC baseline:      {mid_baseline_verdict}")
    logger.info(f"  mid500_R (L22):       {R_mid500.mean():.4f}")
    logger.info(f"  Cohen's d top vs mid: {kohens_d_top_mid:.4f}")
    logger.info(f"  Benign overlap:       {benign_overlap_verdict} ({benign_overlap_fraction:.3f})")
    logger.info(f"  |cos| L22 top500:     {cos_enc_top500_L22_mean_abs:.5f} (iter3: 0.01853)")
    logger.info(f"  |cos| L20 top500:     {cos_enc_top500_L20_mean_abs:.5f} (iter3: 0.02086)")
    logger.info("=" * 60)

    return method_out_path, analysis_path


if __name__ == "__main__":
    method_out_path, analysis_path = main()
    logger.info("Done.")
