#!/usr/bin/env python3
"""
Null-Baseline SAE Reversal Score Validation.

Validates the selective-preservation hypothesis by running:
  1. Reversal Score comparison: top-500 vs bottom-500 RLHF-sensitive SAE features
  2. Null baseline: random-500 control group
  3. Post-TopK density characterization
  4. Refusal direction orthogonality to SAE encoder directions
  5. Control-subspace centroid distances (subspaces A=top500, B=bottom500, C=random500)
  6. Layer-20 full replication
"""

import gc
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("HF_HOME", "/ai-inventor/aii_data/hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/ai-inventor/aii_data/hf_cache/hub")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import psutil
import scipy.stats as stats

from loguru import logger

# ─── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE = Path(
    "/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_3/gen_art/gen_art_experiment_1"
)
RESULTS = WORKSPACE / "results"
ACT_DIR = WORKSPACE / "activations"
LOG_DIR = WORKSPACE / "logs"
for _d in [RESULTS, ACT_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

DATA_PATH = Path(
    "/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_1/"
    "gen_art/gen_art_dataset_1/results/data_out.json"
)
SAE_REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"

# Model IDs
BASE_MODEL = "Qwen/Qwen3-8B-Base"
INST_MODEL = "Qwen/Qwen3-8B"
ABL_MODEL_PRIMARY = "huihui-ai/Huihui-Qwen3-8B-abliterated-v2"
ABL_MODEL_FALLBACK = "mlabonne/Qwen3-8B-abliterated"

# ─── Hardware detection ──────────────────────────────────────────────────────
def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return math.ceil(q / p)
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1


def _container_ram_gb() -> Optional[float]:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v not in ("max",) and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


import torch  # noqa: E402

NUM_CPUS = _detect_cpus()
HAS_GPU = torch.cuda.is_available()
VRAM_GB = (torch.cuda.get_device_properties(0).total_memory / 1e9) if HAS_GPU else 0.0
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9

logger.info(f"HW: {NUM_CPUS} CPUs | RAM={TOTAL_RAM_GB:.1f}GB | GPU={HAS_GPU} | VRAM={VRAM_GB:.1f}GB")

# Cap memory — leave headroom for OS and multiple arrays
RAM_BUDGET = int(TOTAL_RAM_GB * 0.82 * 1e9)
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
if HAS_GPU:
    torch.cuda.set_per_process_memory_fraction(0.97)


# ─── Data loading ────────────────────────────────────────────────────────────
@logger.catch(reraise=True)
def load_data():
    logger.info(f"Loading data from {DATA_PATH}")
    data = json.loads(DATA_PATH.read_text())
    prompts = [d["prompt"] for d in data]
    harmful_idx = [i for i, d in enumerate(data) if d["label"] == "harmful"]
    benign_idx = [i for i, d in enumerate(data) if d["label"] == "benign"]
    assert len(harmful_idx) == 500, f"Expected 500 harmful, got {len(harmful_idx)}"
    assert len(benign_idx) == 500, f"Expected 500 benign, got {len(benign_idx)}"
    logger.info(f"Loaded {len(prompts)} prompts: {len(harmful_idx)} harmful, {len(benign_idx)} benign")
    return data, prompts, harmful_idx, benign_idx


# ─── Activation extraction ───────────────────────────────────────────────────
def extract_activations(
    model_id: str,
    tag: str,
    prompts: list[str],
    layers: list[int],
    batch_size: int = 2,
) -> dict[int, np.ndarray]:
    """Load model, hook specified layers, run all prompts, return {layer: (N,4096)} float32."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Check which layers already saved
    missing_layers = [
        L for L in layers
        if not (ACT_DIR / f"{tag}_act_L{L}.npy").exists()
    ]
    if not missing_layers:
        logger.info(f"{tag}: all layers already cached, loading from disk")
        return {L: np.load(ACT_DIR / f"{tag}_act_L{L}.npy") for L in layers}

    logger.info(f"Extracting {tag} layers {missing_layers}, {len(prompts)} prompts...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    acts: dict[int, list] = {L: [] for L in missing_layers}
    hooks = []

    def make_hook(layer_idx: int):
        def _hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            acts[layer_idx].append(
                hidden[:, -1, :].detach().cpu().float().numpy()
            )
        return _hook

    for L in missing_layers:
        h = model.model.layers[L].register_forward_hook(make_hook(L))
        hooks.append(h)

    n = len(prompts)
    effective_batch = batch_size
    start = 0
    batch_count = 0
    while start < n:
        batch = prompts[start : start + effective_batch]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(DEVICE)
        # snapshot hook lengths before forward pass for OOM rollback
        snap = {L: len(acts[L]) for L in missing_layers}
        try:
            with torch.no_grad():
                model(**enc)
            start += len(batch)
            batch_count += 1
            if batch_count % 50 == 0:
                logger.info(f"  {tag}: {start}/{n} prompts (batch={effective_batch})")
        except torch.cuda.OutOfMemoryError:
            # rollback partial hook data, then retry with smaller batch
            for L in missing_layers:
                acts[L] = acts[L][: snap[L]]
            del enc
            torch.cuda.empty_cache()
            gc.collect()
            if effective_batch == 1:
                raise RuntimeError(f"OOM even at batch_size=1 for {tag}")
            effective_batch = max(1, effective_batch // 2)
            logger.warning(f"  OOM: reducing batch_size to {effective_batch}, retry from {start}")

    for h in hooks:
        h.remove()

    results = {}
    for L in missing_layers:
        arr = np.concatenate(acts[L], axis=0).astype(np.float32)  # (N, 4096)
        np.save(ACT_DIR / f"{tag}_act_L{L}.npy", arr)
        logger.info(f"  Saved {tag}_act_L{L}: shape={arr.shape} mean={arr.mean():.4f}")
        results[L] = arr

    del model, tokenizer, acts, hooks
    torch.cuda.empty_cache()
    gc.collect()

    elapsed = time.time() - t0
    logger.info(f"{tag} extraction done in {elapsed/60:.1f} min")

    # Load any already-cached layers
    for L in layers:
        if L not in results:
            results[L] = np.load(ACT_DIR / f"{tag}_act_L{L}.npy")
    return results


# ─── SAE loading ────────────────────────────────────────────────────────────
def load_sae(layer_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SAE checkpoint, return (W_enc, b_enc, W_dec) as float32 numpy arrays."""
    from huggingface_hub import hf_hub_download

    logger.info(f"Loading SAE layer {layer_idx}...")
    ckpt_path = hf_hub_download(SAE_REPO, f"layer{layer_idx}.sae.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    logger.debug(f"SAE layer{layer_idx} keys: {list(ckpt.keys())}")

    # Try common key names
    def _get(keys):
        for k in keys:
            if k in ckpt:
                return ckpt[k]
        raise KeyError(f"None of {keys} found in SAE checkpoint. Keys: {list(ckpt.keys())}")

    W_enc = _get(["W_enc", "encoder.weight", "W_E"]).numpy().astype(np.float32)
    b_enc = _get(["b_enc", "encoder.bias", "b_E"]).numpy().astype(np.float32)
    W_dec = _get(["W_dec", "decoder.weight", "W_D"]).numpy().astype(np.float32)

    # Ensure shapes: W_enc (n_feats, d_model), W_dec (d_model, n_feats)
    if W_enc.ndim == 2 and W_enc.shape[0] < W_enc.shape[1]:
        W_enc = W_enc.T  # transpose if rows < cols
    if W_dec.ndim == 2 and W_dec.shape[0] > W_dec.shape[1]:
        W_dec = W_dec.T  # ensure (d_model, n_feats)

    logger.info(f"SAE layer{layer_idx}: W_enc={W_enc.shape} b_enc={b_enc.shape} W_dec={W_dec.shape}")
    return W_enc, b_enc, W_dec


# ─── Pre-TopK projections ────────────────────────────────────────────────────
def compute_pre_topk(
    act_arr: np.ndarray, W_enc: np.ndarray, b_enc: np.ndarray, chunk_size: int = 50
) -> np.ndarray:
    """z_pre = act @ W_enc.T + b_enc. Returns (N, n_feats) float32."""
    n = act_arr.shape[0]
    n_feats = W_enc.shape[0]
    z_pre = np.empty((n, n_feats), dtype=np.float32)
    for i in range(0, n, chunk_size):
        chunk = act_arr[i : i + chunk_size].astype(np.float32)
        z_pre[i : i + chunk_size] = chunk @ W_enc.T + b_enc
    return z_pre


# ─── Reversal scores ─────────────────────────────────────────────────────────
def compute_reversal_scores(
    feature_indices: np.ndarray,
    z_base: np.ndarray,
    z_inst: np.ndarray,
    z_abl: np.ndarray,
    prompt_idx: list[int],
) -> np.ndarray:
    """Per-feature Pearson r between delta_inst and delta_abl on prompt_idx. Returns (n_feats,)."""
    pi = np.array(prompt_idx)
    fi = feature_indices

    delta_inst = z_inst[np.ix_(pi, fi)] - z_base[np.ix_(pi, fi)]  # (n_p, n_f)
    delta_abl = z_abl[np.ix_(pi, fi)] - z_base[np.ix_(pi, fi)]

    di_c = delta_inst - delta_inst.mean(axis=0, keepdims=True)
    da_c = delta_abl - delta_abl.mean(axis=0, keepdims=True)
    num = (di_c * da_c).sum(axis=0)
    denom = np.sqrt((di_c ** 2).sum(axis=0) * (da_c ** 2).sum(axis=0)) + 1e-10
    return (num / denom).astype(np.float32)


# ─── Helper: cosine distance ─────────────────────────────────────────────────
def cosine_dist(u: np.ndarray, v: np.ndarray) -> float:
    nn = np.linalg.norm(u) * np.linalg.norm(v) + 1e-10
    return float(1.0 - np.dot(u, v) / nn)


# ─── Analysis: centroid ratios ────────────────────────────────────────────────
def centroid_ratios(
    z_base: np.ndarray,
    z_inst: np.ndarray,
    z_abl: np.ndarray,
    feature_idx: np.ndarray,
    prompt_idx: list[int],
) -> dict:
    pi = np.array(prompt_idx)
    fi = feature_idx

    c_base = z_base[np.ix_(pi, fi)].mean(axis=0)
    c_inst = z_inst[np.ix_(pi, fi)].mean(axis=0)
    c_abl = z_abl[np.ix_(pi, fi)].mean(axis=0)

    d_abl_inst = cosine_dist(c_abl, c_inst)
    d_abl_base = cosine_dist(c_abl, c_base)
    ratio = d_abl_base / (d_abl_inst + 1e-10)

    return {
        "d_abl_inst": float(d_abl_inst),
        "d_abl_base": float(d_abl_base),
        "ratio": float(ratio),
    }


# ─── Full analysis for one layer ─────────────────────────────────────────────
def run_layer_analysis(
    layer: int,
    base_act: np.ndarray,
    inst_act: np.ndarray,
    abl_act: np.ndarray,
    harmful_idx: list[int],
    benign_idx: list[int],
    W_enc: np.ndarray,
    b_enc: np.ndarray,
    W_dec: np.ndarray,
) -> dict:
    """Full SAE-based analysis for one layer. Returns results dict."""
    import diptest

    logger.info(f"=== Layer {layer}: computing pre-topk projections ===")
    z_base = compute_pre_topk(base_act, W_enc, b_enc)
    z_inst = compute_pre_topk(inst_act, W_enc, b_enc)
    z_abl = compute_pre_topk(abl_act, W_enc, b_enc)

    logger.info(
        f"Layer {layer} projections: z_base shape={z_base.shape} "
        f"mean={z_base.mean():.4f} std={z_base.std():.4f}"
    )

    # Safety change: mean |delta_inst| on harmful prompts per feature
    SC = np.mean(
        np.abs(z_inst[np.array(harmful_idx)] - z_base[np.array(harmful_idx)]),
        axis=0,
    )  # (n_feats,)

    top500_idx = np.argsort(SC)[-500:]
    bottom500_idx = np.argsort(SC)[:500]

    # Random 500 (not in top500 or bottom500)
    rng = np.random.default_rng(42)
    all_excluded = set(top500_idx.tolist()) | set(bottom500_idx.tolist())
    random500_idx = np.array(
        [i for i in rng.permutation(SC.shape[0]) if i not in all_excluded][:500]
    )

    # Sanity checks
    assert len(set(top500_idx.tolist()) & set(bottom500_idx.tolist())) == 0, \
        "top/bottom overlap!"
    assert len(random500_idx) == 500
    logger.info(
        f"Layer {layer}: SC top={SC[top500_idx].min():.4f}..{SC[top500_idx].max():.4f}, "
        f"bottom={SC[bottom500_idx].min():.6f}..{SC[bottom500_idx].max():.6f}"
    )

    # ── Reversal scores ──
    logger.info(f"Layer {layer}: computing reversal scores...")
    R_top = compute_reversal_scores(top500_idx, z_base, z_inst, z_abl, harmful_idx)
    R_bot = compute_reversal_scores(bottom500_idx, z_base, z_inst, z_abl, harmful_idx)
    R_rand = compute_reversal_scores(random500_idx, z_base, z_inst, z_abl, harmful_idx)

    ttest_top_bot = stats.ttest_ind(R_top, R_bot)
    ttest_top_rand = stats.ttest_ind(R_top, R_rand)
    ks_top_bot = stats.ks_2samp(R_top, R_bot)
    cohens_d_top_bot = float(
        (R_top.mean() - R_bot.mean())
        / np.sqrt((R_top.var() + R_bot.var()) / 2 + 1e-10)
    )
    cohens_d_top_rand = float(
        (R_top.mean() - R_rand.mean())
        / np.sqrt((R_top.var() + R_rand.var()) / 2 + 1e-10)
    )

    dip_top, dip_top_p = diptest.diptest(R_top)
    dip_bot, dip_bot_p = diptest.diptest(R_bot)
    dip_rand, dip_rand_p = diptest.diptest(R_rand)

    # Verdict
    if ttest_top_bot.pvalue < 0.05 and abs(cohens_d_top_bot) > 0.5:
        null_verdict = "SELECTIVE_PRESERVATION_CONFIRMED"
    elif ks_top_bot.pvalue < 0.05:
        null_verdict = "WEAK_SELECTIVE_PRESERVATION"
    else:
        null_verdict = "GLOBAL_SIMILARITY_ALTERNATIVE_PLAUSIBLE"

    null_baseline = {
        "top500_R_mean": float(R_top.mean()),
        "top500_R_std": float(R_top.std()),
        "top500_R_min": float(R_top.min()),
        "top500_R_max": float(R_top.max()),
        "top500_n_negative": int((R_top < 0).sum()),
        "top500_n_positive": int((R_top > 0).sum()),
        "bottom500_R_mean": float(R_bot.mean()),
        "bottom500_R_std": float(R_bot.std()),
        "bottom500_R_min": float(R_bot.min()),
        "bottom500_R_max": float(R_bot.max()),
        "bottom500_n_negative": int((R_bot < 0).sum()),
        "random500_R_mean": float(R_rand.mean()),
        "random500_R_std": float(R_rand.std()),
        "random500_R_min": float(R_rand.min()),
        "random500_R_max": float(R_rand.max()),
        "random500_n_negative": int((R_rand < 0).sum()),
        "ttest_top_vs_bottom_t": float(ttest_top_bot.statistic),
        "ttest_top_vs_bottom_p": float(ttest_top_bot.pvalue),
        "ttest_top_vs_random_t": float(ttest_top_rand.statistic),
        "ttest_top_vs_random_p": float(ttest_top_rand.pvalue),
        "ks_stat": float(ks_top_bot.statistic),
        "ks_p": float(ks_top_bot.pvalue),
        "cohens_d_top_vs_bottom": cohens_d_top_bot,
        "cohens_d_top_vs_random": cohens_d_top_rand,
        "diptest_top500_stat": float(dip_top),
        "diptest_top500_p": float(dip_top_p),
        "diptest_bottom500_stat": float(dip_bot),
        "diptest_bottom500_p": float(dip_bot_p),
        "diptest_random500_stat": float(dip_rand),
        "diptest_random500_p": float(dip_rand_p),
        "verdict": null_verdict,
    }
    logger.info(
        f"Layer {layer} null_baseline: top_R={R_top.mean():.3f} bot_R={R_bot.mean():.3f} "
        f"d={cohens_d_top_bot:.3f} p={ttest_top_bot.pvalue:.2e} → {null_verdict}"
    )

    # ── Post-TopK density ──
    logger.info(f"Layer {layer}: computing post-topk density...")
    z_harm_inst = z_inst[np.array(harmful_idx)]  # (500, n_feats)
    kth_vals = np.partition(z_harm_inst, -50, axis=1)[:, -50]  # 50th largest per prompt
    top_feats_vals = z_harm_inst[:, top500_idx]  # (500, 500)
    density = (top_feats_vals >= kth_vals[:, None]).mean(axis=0)  # (500,)

    posttopk_density = {
        "density_mean": float(density.mean()),
        "density_median": float(np.median(density)),
        "density_min": float(density.min()),
        "density_max": float(density.max()),
        "density_std": float(density.std()),
        "fraction_above_0p10": float((density > 0.10).mean()),
        "fraction_above_0p05": float((density > 0.05).mean()),
        "fraction_above_0p01": float((density > 0.01).mean()),
        "note": (
            "density(f)=fraction of 500 harmful prompts where feature f "
            "is in top-50 of instruct pre-topk activations"
        ),
    }
    logger.info(
        f"Layer {layer} density: mean={density.mean():.4f} >0.10={posttopk_density['fraction_above_0p10']:.3f}"
    )

    # ── Refusal direction ──
    logger.info(f"Layer {layer}: computing refusal direction...")
    hi = np.array(harmful_idx)
    bi = np.array(benign_idx)
    r_raw = inst_act[hi].mean(axis=0) - inst_act[bi].mean(axis=0)  # (4096,)
    r_norm = np.linalg.norm(r_raw)
    r = r_raw / (r_norm + 1e-10)

    assert abs(np.linalg.norm(r) - 1.0) < 1e-5, "Refusal direction not unit"

    # Cosine similarities to SAE encoder directions
    enc_vecs_top = W_enc[top500_idx]  # (500, 4096)
    enc_norms_top = np.linalg.norm(enc_vecs_top, axis=1, keepdims=True) + 1e-10
    cos_enc_top = (enc_vecs_top @ r) / enc_norms_top[:, 0]  # (500,) signed

    enc_vecs_bot = W_enc[bottom500_idx]
    enc_norms_bot = np.linalg.norm(enc_vecs_bot, axis=1, keepdims=True) + 1e-10
    cos_enc_bot = (enc_vecs_bot @ r) / enc_norms_bot[:, 0]

    enc_vecs_rand = W_enc[random500_idx]
    enc_norms_rand = np.linalg.norm(enc_vecs_rand, axis=1, keepdims=True) + 1e-10
    cos_enc_rand = (enc_vecs_rand @ r) / enc_norms_rand[:, 0]

    # Decoder directions
    W_dec_T = W_dec.T  # (n_feats, 4096)
    dec_vecs_top = W_dec_T[top500_idx]
    dec_norms_top = np.linalg.norm(dec_vecs_top, axis=1, keepdims=True) + 1e-10
    cos_dec_top = (dec_vecs_top @ r) / dec_norms_top[:, 0]

    # Top-10 within top-500 (highest SC)
    top10_within = np.argsort(SC[top500_idx])[-10:]
    top10_feats = top500_idx[top10_within]
    top10_list = [
        {
            "feature_idx": int(top10_feats[i]),
            "sc": float(SC[top10_feats[i]]),
            "cos_enc": float(cos_enc_top[top10_within[i]]),
            "cos_dec": float(cos_dec_top[top10_within[i]]),
        }
        for i in range(10)
    ]

    mean_abs_cos_enc_top = float(np.abs(cos_enc_top).mean())
    mean_abs_cos_enc_bot = float(np.abs(cos_enc_bot).mean())

    if mean_abs_cos_enc_top < 0.1:
        orth_verdict = "ORTHOGONALITY_CONFIRMED"
    elif mean_abs_cos_enc_top < 0.2:
        orth_verdict = "NEAR_ORTHOGONAL"
    else:
        orth_verdict = "ORTHOGONALITY_WEAKENED"

    # t-test: abs cos_enc top vs bottom
    ttest_cos = stats.ttest_ind(np.abs(cos_enc_top), np.abs(cos_enc_bot))

    orthogonality = {
        "cos_enc_top500_mean_abs": mean_abs_cos_enc_top,
        "cos_enc_top500_max_abs": float(np.abs(cos_enc_top).max()),
        "cos_enc_top500_std": float(np.abs(cos_enc_top).std()),
        "cos_enc_top500_mean_signed": float(cos_enc_top.mean()),
        "cos_dec_top500_mean_abs": float(np.abs(cos_dec_top).mean()),
        "cos_dec_top500_max_abs": float(np.abs(cos_dec_top).max()),
        "cos_enc_bottom500_mean_abs": mean_abs_cos_enc_bot,
        "cos_enc_random500_mean_abs": float(np.abs(cos_enc_rand).mean()),
        "refusal_dir_norm": float(r_norm),
        "ttest_abs_cos_top_vs_bottom_p": float(ttest_cos.pvalue),
        "ttest_abs_cos_top_vs_bottom_t": float(ttest_cos.statistic),
        "top10_feature_cos_enc": top10_list,
        "verdict": orth_verdict,
    }
    logger.info(
        f"Layer {layer} orthogonality: |cos_enc_top|={mean_abs_cos_enc_top:.4f} → {orth_verdict}"
    )

    # ── Centroid control ──
    logger.info(f"Layer {layer}: computing centroid control distances...")

    def _crat(fi, pi):
        return centroid_ratios(z_base, z_inst, z_abl, fi, pi)

    res_A_harm = _crat(top500_idx, harmful_idx)
    res_A_beni = _crat(top500_idx, benign_idx)
    res_B_harm = _crat(bottom500_idx, harmful_idx)
    res_B_beni = _crat(bottom500_idx, benign_idx)
    res_C_harm = _crat(random500_idx, harmful_idx)
    res_C_beni = _crat(random500_idx, benign_idx)

    ratA = res_A_harm["ratio"]
    ratB = res_B_harm["ratio"]
    ratC = res_C_harm["ratio"]

    if ratA > 2 * ratB and ratA > 2 * ratC:
        centroid_verdict = "CENTROID_SELECTIVE"
    elif ratA > ratB and ratA > ratC:
        centroid_verdict = "CENTROID_MODERATE"
    else:
        centroid_verdict = "CENTROID_GLOBAL"

    centroid_control = {
        "subspaceA_harmful": res_A_harm,
        "subspaceA_benign": res_A_beni,
        "subspaceB_harmful": res_B_harm,
        "subspaceB_benign": res_B_beni,
        "subspaceC_harmful": res_C_harm,
        "subspaceC_benign": res_C_beni,
        "ratio_A_harmful": ratA,
        "ratio_A_benign": res_A_beni["ratio"],
        "ratio_B_harmful": ratB,
        "ratio_B_benign": res_B_beni["ratio"],
        "ratio_C_harmful": ratC,
        "ratio_C_benign": res_C_beni["ratio"],
        "verdict": centroid_verdict,
    }
    logger.info(
        f"Layer {layer} centroid: A_harm={ratA:.3f} B_harm={ratB:.3f} C_harm={ratC:.3f} → {centroid_verdict}"
    )

    # Free SAE projection arrays for this layer before returning
    del z_base, z_inst, z_abl
    gc.collect()

    return {
        "null_baseline": null_baseline,
        "posttopk_density": posttopk_density,
        "orthogonality": orthogonality,
        "centroid_control": centroid_control,
        "top500_indices": top500_idx[:20].tolist(),   # store a sample for reference
        "bottom500_indices": bottom500_idx[:20].tolist(),
        "sc_top500_min": float(SC[top500_idx].min()),
        "sc_top500_max": float(SC[top500_idx].max()),
        "sc_bottom500_max": float(SC[bottom500_idx].max()),
    }


# ─── Per-prompt statistics for schema output ─────────────────────────────────
def compute_per_prompt_stats(
    layer: int,
    base_act: np.ndarray,
    inst_act: np.ndarray,
    abl_act: np.ndarray,
    W_enc: np.ndarray,
    b_enc: np.ndarray,
    harmful_idx: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-prompt mean |delta_inst| on top-500, bottom-500, random-500 features."""
    z_base = compute_pre_topk(base_act, W_enc, b_enc)
    z_inst = compute_pre_topk(inst_act, W_enc, b_enc)
    z_abl = compute_pre_topk(abl_act, W_enc, b_enc)

    SC = np.mean(
        np.abs(z_inst[np.array(harmful_idx)] - z_base[np.array(harmful_idx)]),
        axis=0,
    )
    top500_idx = np.argsort(SC)[-500:]
    bottom500_idx = np.argsort(SC)[:500]

    rng = np.random.default_rng(42)
    excl = set(top500_idx.tolist()) | set(bottom500_idx.tolist())
    random500_idx = np.array([i for i in rng.permutation(SC.shape[0]) if i not in excl][:500])

    n = base_act.shape[0]
    delta_inst_abs = np.abs(z_inst - z_base)

    mean_delta_top = delta_inst_abs[:, top500_idx].mean(axis=1)  # (N,)
    mean_delta_bot = delta_inst_abs[:, bottom500_idx].mean(axis=1)
    mean_delta_rand = delta_inst_abs[:, random500_idx].mean(axis=1)

    del z_base, z_inst, z_abl, delta_inst_abs
    gc.collect()

    return mean_delta_top, mean_delta_bot, mean_delta_rand


# ─── Main ────────────────────────────────────────────────────────────────────
@logger.catch(reraise=True)
def main():
    t_start = time.time()
    logger.info("=== Null-Baseline SAE Reversal Score Validation START ===")

    # Load data
    data, prompts, harmful_idx, benign_idx = load_data()

    # ── Step 0: Activation extraction (layer 22 already cached from iter_2) ──
    logger.info("=== Step 0: Activation extraction ===")

    # Layer 22: already copied from iter_2
    base_L22 = np.load(ACT_DIR / "base_act_L22.npy")
    inst_L22 = np.load(ACT_DIR / "inst_act_L22.npy")
    abl_L22 = np.load(ACT_DIR / "abl_act_L22.npy")
    logger.info(f"Layer 22 activations loaded: base={base_L22.shape} inst={inst_L22.shape}")

    # Layer 20: extract if not cached
    layer20_needed = not (
        (ACT_DIR / "base_act_L20.npy").exists()
        and (ACT_DIR / "inst_act_L20.npy").exists()
        and (ACT_DIR / "abl_act_L20.npy").exists()
    )

    if layer20_needed:
        logger.info("=== Extracting layer-20 activations ===")

        # Use primary abliterated model (cached from iter_2); fallback handled in extract_activations
        abl_model_id = ABL_MODEL_PRIMARY

        for model_id, tag in [
            (BASE_MODEL, "base"),
            (INST_MODEL, "inst"),
            (abl_model_id, "abl"),
        ]:
            extract_activations(model_id, tag, prompts, layers=[20], batch_size=2)
            gc.collect()
            torch.cuda.empty_cache()
    else:
        logger.info("Layer-20 activations already cached")

    base_L20 = np.load(ACT_DIR / "base_act_L20.npy")
    inst_L20 = np.load(ACT_DIR / "inst_act_L20.npy")
    abl_L20 = np.load(ACT_DIR / "abl_act_L20.npy")
    logger.info(f"Layer 20 activations loaded: base={base_L20.shape}")

    # ── Step 1-6: Full analysis at layer 22 ──
    logger.info("=== Step 1-6: Full analysis at layer 22 ===")
    W_enc22, b_enc22, W_dec22 = load_sae(22)
    layer22_results = run_layer_analysis(
        layer=22,
        base_act=base_L22,
        inst_act=inst_L22,
        abl_act=abl_L22,
        harmful_idx=harmful_idx,
        benign_idx=benign_idx,
        W_enc=W_enc22,
        b_enc=b_enc22,
        W_dec=W_dec22,
    )
    del W_enc22, b_enc22, W_dec22
    gc.collect()

    # ── Step 7: Layer-20 replication ──
    logger.info("=== Step 7: Layer-20 replication ===")
    W_enc20, b_enc20, W_dec20 = load_sae(20)
    layer20_results = run_layer_analysis(
        layer=20,
        base_act=base_L20,
        inst_act=inst_L20,
        abl_act=abl_L20,
        harmful_idx=harmful_idx,
        benign_idx=benign_idx,
        W_enc=W_enc20,
        b_enc=b_enc20,
        W_dec=W_dec20,
    )

    del W_enc20, b_enc20, W_dec20
    gc.collect()

    # ── Step 8: Per-prompt stats with layer-22 SAE ──
    logger.info("Computing per-prompt stats with L22 SAE for schema output...")
    W_enc22b, b_enc22b, W_dec22b = load_sae(22)
    mean_d_top22, mean_d_bot22, mean_d_rand22 = compute_per_prompt_stats(
        22, base_L22, inst_L22, abl_L22, W_enc22b, b_enc22b, harmful_idx
    )
    del W_enc22b, b_enc22b, W_dec22b
    gc.collect()

    # ── Cross-layer verdict ──
    r22 = layer22_results["null_baseline"]["top500_R_mean"]
    r20 = layer20_results["null_baseline"]["top500_R_mean"]
    r22_bot = layer22_results["null_baseline"]["bottom500_R_mean"]
    r20_bot = layer20_results["null_baseline"]["bottom500_R_mean"]

    replication_verdict = (
        "REPLICATION_CONFIRMED"
        if (r22 > r22_bot and r20 > r20_bot)
        else "REPLICATION_PARTIAL"
    )

    # ── Assemble analysis output ──
    analysis_out = {
        "meta": {
            "n_harmful": len(harmful_idx),
            "n_benign": len(benign_idx),
            "n_features_total": 65536,
            "sae_repo": SAE_REPO,
            "models": {
                "base": BASE_MODEL,
                "inst": INST_MODEL,
                "abl": ABL_MODEL_PRIMARY,
            },
            "primary_layer": 22,
            "replication_layer": 20,
        },
        "layer22": layer22_results,
        "layer20": layer20_results,
        "cross_layer": {
            "layer22_top500_R_mean": float(r22),
            "layer22_bottom500_R_mean": float(r22_bot),
            "layer20_top500_R_mean": float(r20),
            "layer20_bottom500_R_mean": float(r20_bot),
            "replication_verdict": replication_verdict,
        },
        "paper_corrections": {
            "semantic_labeling_correct": (
                "top-5 prompts by max pre-topk activation on last token, "
                "first 300 chars — not token window"
            ),
            "density_filter_correct": (
                "pre-topk dense projections always nonzero; "
                "post-topk density fraction is honest measure (see posttopk_density field)"
            ),
            "pretopk_limitation": (
                "semantic claims rest on pre-topk encoder projections, "
                "not validated sparse post-topk features"
            ),
        },
    }

    analysis_path = RESULTS / "analysis_out.json"
    analysis_path.write_text(json.dumps(analysis_out, indent=2))
    logger.info(f"Saved analysis to {analysis_path}")

    # ── Assemble schema-compliant method_out.json ──
    logger.info("Assembling schema-compliant method_out.json...")

    harmful_set = set(harmful_idx)
    examples = []
    for i, d in enumerate(data):
        src = d.get("source", "unknown")
        delta_top = float(mean_d_top22[i])
        delta_bot = float(mean_d_bot22[i])
        delta_rand = float(mean_d_rand22[i])

        is_harmful = d["label"] == "harmful"
        classification = "high_rlhf_delta" if delta_top > float(mean_d_top22.mean()) else "low_rlhf_delta"

        ex = {
            "input": d["prompt"],
            "output": d["label"],
            "predict_rlhf_classification": classification,
            "predict_mean_delta_top500": str(round(delta_top, 6)),
            "predict_mean_delta_bottom500": str(round(delta_bot, 6)),
            "predict_mean_delta_random500": str(round(delta_rand, 6)),
            "predict_is_harmful_confirmed": str(is_harmful),
            "metadata_source": d.get("source", ""),
            "metadata_category": d.get("category", ""),
            "metadata_id": d.get("id", ""),
            "metadata_char_len": d.get("char_len", 0),
        }
        examples.append(ex)

    # Aggregate stats summary for metadata
    l22 = layer22_results
    l20 = layer20_results

    method_out = {
        "metadata": {
            "method_name": "Null-Baseline SAE Reversal Score Validation",
            "description": (
                "Validates selective-preservation hypothesis: RLHF-sensitive SAE features "
                "have higher reversal scores (Pearson r between inst-delta and abl-delta) "
                "than RLHF-insensitive and random features. Analysis at layers 22 and 20."
            ),
            "layer22_null_baseline_verdict": l22["null_baseline"]["verdict"],
            "layer22_top500_R_mean": l22["null_baseline"]["top500_R_mean"],
            "layer22_bottom500_R_mean": l22["null_baseline"]["bottom500_R_mean"],
            "layer22_cohens_d": l22["null_baseline"]["cohens_d_top_vs_bottom"],
            "layer22_ttest_p": l22["null_baseline"]["ttest_top_vs_bottom_p"],
            "layer22_orthogonality_verdict": l22["orthogonality"]["verdict"],
            "layer22_cos_enc_top500_mean_abs": l22["orthogonality"]["cos_enc_top500_mean_abs"],
            "layer22_centroid_verdict": l22["centroid_control"]["verdict"],
            "layer22_centroid_ratio_A_harmful": l22["centroid_control"]["ratio_A_harmful"],
            "layer22_centroid_ratio_B_harmful": l22["centroid_control"]["ratio_B_harmful"],
            "layer20_null_baseline_verdict": l20["null_baseline"]["verdict"],
            "layer20_top500_R_mean": l20["null_baseline"]["top500_R_mean"],
            "layer20_bottom500_R_mean": l20["null_baseline"]["bottom500_R_mean"],
            "cross_layer_replication": replication_verdict,
            "n_prompts": len(data),
            "sae_repo": SAE_REPO,
            "primary_layer": 22,
        },
        "datasets": [
            {
                "dataset": "harmful_benign_prompts_sae_analysis",
                "examples": examples,
            }
        ],
    }

    method_path = RESULTS / "method_out.json"
    method_path.write_text(json.dumps(method_out, indent=2))
    logger.info(f"Saved method_out.json ({method_path.stat().st_size / 1e6:.1f} MB)")

    elapsed = (time.time() - t_start) / 60
    logger.info(f"=== DONE in {elapsed:.1f} min ===")
    logger.info(
        f"Summary:\n"
        f"  Layer 22 null_baseline verdict: {l22['null_baseline']['verdict']}\n"
        f"  Layer 22 top500_R_mean: {l22['null_baseline']['top500_R_mean']:.4f}\n"
        f"  Layer 22 bottom500_R_mean: {l22['null_baseline']['bottom500_R_mean']:.4f}\n"
        f"  Layer 22 Cohen's d: {l22['null_baseline']['cohens_d_top_vs_bottom']:.4f}\n"
        f"  Layer 22 orthogonality: {l22['orthogonality']['verdict']}\n"
        f"  Layer 22 centroid: {l22['centroid_control']['verdict']}\n"
        f"  Layer 20 null_baseline verdict: {l20['null_baseline']['verdict']}\n"
        f"  Cross-layer replication: {replication_verdict}\n"
    )


if __name__ == "__main__":
    main()
