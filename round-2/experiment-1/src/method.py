#!/usr/bin/env python3
"""
Qwen3-8B SAE Reversal Score Analysis

Mechanistic interpretability pipeline comparing base, instruct, and abliterated
Qwen3-8B activations through frozen Qwen-Scope TopK SAEs.

Computes per-feature Reversal_scores (Pearson correlation of RLHF vs abliteration
change vectors), tests bimodality, measures centroid distances, and validates
semantic labels via cross-family LLM labeler.
"""

import gc
import json
import math
import os
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Point HF cache to large MFS network filesystem (35GB root is too small for 3×16GB models)
os.environ.setdefault("HF_HOME", "/ai-inventor/aii_data/hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/ai-inventor/aii_data/hf_cache/hub")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

import numpy as np
import psutil
import scipy.stats as stats
from loguru import logger
from scipy.spatial.distance import cosine
from scipy.stats import fisher_exact

# ============================================================
# PATHS & DIRECTORIES
# ============================================================
WORKSPACE = Path("/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_2/gen_art/gen_art_experiment_1")
RESULTS = WORKSPACE / "results"
ACTS_DIR = WORKSPACE / "activations"
LOGS_DIR = WORKSPACE / "logs"
for _d in [RESULTS, ACTS_DIR, LOGS_DIR]:
    _d.mkdir(exist_ok=True)

# ============================================================
# LOGGING
# ============================================================
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ============================================================
# HARDWARE DETECTION
# ============================================================
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
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()

import torch  # noqa: E402 (after path setup)

HAS_GPU = torch.cuda.is_available()
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_GPU else 0.0
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9

logger.info(f"HW: {NUM_CPUS} CPUs | RAM={TOTAL_RAM_GB:.1f}GB | GPU={HAS_GPU} | VRAM={VRAM_GB:.1f}GB")

# Memory limits (cgroup-aware)
RAM_BUDGET_GB = TOTAL_RAM_GB * 0.85
RAM_BUDGET = int(RAM_BUDGET_GB * 1e9)
_avail = psutil.virtual_memory().available
RAM_BUDGET = min(RAM_BUDGET, int(_avail * 0.90))
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))
logger.info(f"RAM budget: {RAM_BUDGET / 1e9:.1f}GB")

if HAS_GPU:
    _free, _total = torch.cuda.mem_get_info(0)
    torch.cuda.set_per_process_memory_fraction(0.90)
    logger.info(f"VRAM: {_free / 1e9:.1f}GB free / {_total / 1e9:.1f}GB total")

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_PATH = Path(
    "/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/results/data_out.json"
)

MODEL_IDS = {
    "base": "Qwen/Qwen3-8B-Base",
    "instruct": "Qwen/Qwen3-8B",
    "abliterated": "huihui-ai/Huihui-Qwen3-8B-abliterated-v2",
}
ABLITERATED_FALLBACK = "mlabonne/Qwen3-8B-abliterated"

SAE_REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
K_TOPK = 50
N_LAYERS = 36
D_MODEL = 4096
N_FEATURES = 65536

# Literature-informed pilot layers (saves time vs all-36 scan)
PILOT_LAYERS = [14, 20, 22]
PILOT_N = 50  # harmful prompts for layer selection

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LABELER_MODEL = "meta-llama/llama-3.1-8b-instruct"

# ============================================================
# BASELINE: KEYWORD-BASED HARM CLASSIFIER
# ============================================================
HARM_KEYWORDS = [
    "kill", "murder", "bomb", "weapon", "poison", "hack", "exploit",
    "illegal", "drug", "violence", "suicide", "abuse", "steal",
    "fraud", "scam", "malware", "ransomware", "phishing", "attack",
    "synthesize", "harmful", "threat", "terror", "extremist",
    "child abuse", "instructions for", "how to make", "how to get away",
    "bypass", "jailbreak", "ignore your instructions", "ignore all",
    "dangerous", "detonate", "explosiv", "bioweapon", "chemical weapon",
    "rape", "torture", "trafficking", "smuggl",
]


def baseline_predict(prompt: str) -> str:
    prompt_lower = prompt.lower()
    score = sum(1 for kw in HARM_KEYWORDS if kw in prompt_lower)
    return "harmful" if score >= 1 else "benign"


# ============================================================
# DATA LOADING
# ============================================================

@logger.catch(reraise=True)
def load_data() -> tuple[list, list, list, np.ndarray, np.ndarray]:
    logger.info(f"Loading data from {DATASET_PATH}")
    data = json.loads(DATASET_PATH.read_text())
    logger.info(f"Loaded {len(data)} examples")

    harmful_mask = np.array([r["label"] == "harmful" for r in data])
    harmful_indices = np.where(harmful_mask)[0]
    benign_indices = np.where(~harmful_mask)[0]

    logger.info(f"Harmful: {len(harmful_indices)}, Benign: {len(benign_indices)}")
    return data, harmful_indices, benign_indices, harmful_mask


# ============================================================
# SAE UTILITIES
# ============================================================

def load_sae(layer_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download and load SAE weights. Returns (W_enc, b_enc, W_dec, b_dec)."""
    from huggingface_hub import hf_hub_download
    sae_file = hf_hub_download(SAE_REPO, f"layer{layer_idx}.sae.pt")
    sae = torch.load(sae_file, map_location="cpu", weights_only=True)

    logger.debug(f"SAE layer{layer_idx} keys: {list(sae.keys())}")
    logger.debug(f"SAE shapes: {[(k, tuple(v.shape)) for k, v in sae.items()]}")

    W_enc = sae["W_enc"].float().numpy()  # (65536, 4096) or (4096, 65536)
    b_enc = sae["b_enc"].float().numpy()  # (65536,)
    W_dec = sae["W_dec"].float().numpy()  # (4096, 65536) or (65536, 4096)
    b_dec = sae["b_dec"].float().numpy()  # (4096,)

    # Shape normalization: ensure W_enc=(N_FEATURES, D_MODEL), W_dec=(D_MODEL, N_FEATURES)
    if W_enc.shape == (D_MODEL, N_FEATURES):
        logger.warning(f"W_enc shape {W_enc.shape}: transposing to ({N_FEATURES}, {D_MODEL})")
        W_enc = W_enc.T
    if W_dec.shape == (N_FEATURES, D_MODEL):
        logger.warning(f"W_dec shape {W_dec.shape}: transposing to ({D_MODEL}, {N_FEATURES})")
        W_dec = W_dec.T

    assert W_enc.shape == (N_FEATURES, D_MODEL), f"W_enc shape {W_enc.shape}"
    assert W_dec.shape == (D_MODEL, N_FEATURES), f"W_dec shape {W_dec.shape}"

    del sae
    gc.collect()
    return W_enc, b_enc, W_dec, b_dec


def sae_project_pretopk(activations: np.ndarray, W_enc: np.ndarray, b_enc: np.ndarray) -> np.ndarray:
    """activations (n, 4096) -> z_pre (n, 65536). Processed in batches to manage RAM."""
    BATCH = 64
    n = len(activations)
    out = np.empty((n, N_FEATURES), dtype=np.float32)
    for start in range(0, n, BATCH):
        batch = activations[start:start + BATCH].astype(np.float32)
        out[start:start + BATCH] = batch @ W_enc.T + b_enc
    return out


def sae_project_posttopk(z_pre: np.ndarray) -> np.ndarray:
    """Apply TopK gate: keep top-K activations per sample."""
    z_post = np.zeros_like(z_pre)
    for i in range(len(z_pre)):
        top_idx = np.argpartition(z_pre[i], -K_TOPK)[-K_TOPK:]
        # Exact sort within top-K for correctness
        top_idx = top_idx[np.argsort(z_pre[i, top_idx])[::-1]]
        z_post[i, top_idx] = z_pre[i, top_idx]
    return z_post


def compute_r2(
    activations: np.ndarray,
    W_enc: np.ndarray,
    b_enc: np.ndarray,
    W_dec: np.ndarray,
    b_dec: np.ndarray,
) -> float:
    """Compute SAE reconstruction R²."""
    z_pre = sae_project_pretopk(activations, W_enc, b_enc)
    z_post = sae_project_posttopk(z_pre)
    reconstruction = (z_post @ W_dec.T) + b_dec
    residual = activations.astype(np.float32) - reconstruction
    ss_res = float((residual ** 2).sum(axis=1).mean())
    a_centered = activations.astype(np.float32) - activations.mean(axis=0)
    ss_tot = float((a_centered ** 2).sum(axis=1).mean())
    return 1.0 - ss_res / (ss_tot + 1e-12)


# ============================================================
# ACTIVATION EXTRACTION
# ============================================================

def extract_multilayer(
    model_id: str,
    prompts: list[str],
    layer_indices: list[int],
    save_path: Path,
) -> dict[int, np.ndarray]:
    """Extract last-token hidden states at multiple layers simultaneously."""
    if save_path.exists():
        logger.info(f"Cache hit: {save_path}")
        npz = np.load(save_path)
        return {int(k): npz[k] for k in npz.files}

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading {model_id} for layers {layer_indices}, {len(prompts)} prompts...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    cache: dict[int, list] = {li: [] for li in layer_indices}
    hooks = []
    for li in layer_indices:
        def _make_hook(idx: int):
            def _hook(module, input, output):
                # transformers 5.x: Qwen3 decoder returns hidden_states tensor directly
                h = output[0] if isinstance(output, (tuple, list)) else output
                cache[idx].append(h[:, -1, :].detach().cpu().float().numpy())
            return _hook
        hooks.append(model.model.layers[li].register_forward_hook(_make_hook(li)))

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            model(**inputs)
            if (i + 1) % 50 == 0:
                logger.info(f"  {i+1}/{len(prompts)} prompts extracted")

    for h in hooks:
        h.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    result = {li: np.concatenate(cache[li], axis=0) for li in layer_indices}
    np.savez_compressed(save_path, **{str(li): arr for li, arr in result.items()})
    logger.info(f"Saved pilot activations to {save_path}")
    return result


def extract_single_layer(
    model_id: str,
    prompts: list[str],
    layer_idx: int,
    save_path: Path,
) -> np.ndarray:
    """Extract last-token hidden state at a single layer for all prompts."""
    if save_path.exists():
        logger.info(f"Cache hit: {save_path}")
        return np.load(save_path)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading {model_id} for layer {layer_idx}, {len(prompts)} prompts...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    acts = []
    hook_out: dict = {}

    def _hook(module, input, output):
        # transformers 5.x: Qwen3 decoder returns hidden_states tensor directly
        h = output[0] if isinstance(output, (tuple, list)) else output
        hook_out["x"] = h[:, -1, :].detach().cpu().float().numpy()

    h = model.model.layers[layer_idx].register_forward_hook(_hook)

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            model(**inputs)
            acts.append(hook_out["x"][0])
            if (i + 1) % 100 == 0:
                logger.info(f"  {i+1}/{len(prompts)} prompts")

    h.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    result = np.stack(acts, axis=0)  # (n, 4096)
    np.save(save_path, result)
    logger.info(f"Saved {save_path}, shape={result.shape}")
    return result


# ============================================================
# CENTROID DISTANCE HELPER
# ============================================================

def centroid_distances(
    inst_sub: np.ndarray,
    abl_sub: np.ndarray,
    base_sub: np.ndarray,
) -> dict:
    c_inst = inst_sub.mean(axis=0)
    c_abl = abl_sub.mean(axis=0)
    c_base = base_sub.mean(axis=0)
    try:
        d_abl_inst_cos = float(cosine(c_abl, c_inst))
        d_abl_base_cos = float(cosine(c_abl, c_base))
        d_inst_base_cos = float(cosine(c_inst, c_base))
    except Exception:
        d_abl_inst_cos = d_abl_base_cos = d_inst_base_cos = float("nan")
    return {
        "d_abl_inst_cosine": d_abl_inst_cos,
        "d_abl_base_cosine": d_abl_base_cos,
        "d_inst_base_cosine": d_inst_base_cos,
        "d_abl_inst_l2": float(np.linalg.norm(c_abl - c_inst)),
        "d_abl_base_l2": float(np.linalg.norm(c_abl - c_base)),
    }


# ============================================================
# MAIN
# ============================================================

@logger.catch(reraise=True)
def main() -> None:
    t_start = time.time()

    # --------------------------------------------------------
    # STEP 0: Load data
    # --------------------------------------------------------
    all_data, harmful_indices, benign_indices, harmful_mask = load_data()
    n_harmful = int(len(harmful_indices))
    n_benign = int(len(benign_indices))
    n_total = len(all_data)

    all_prompts = [r["prompt"] for r in all_data]
    harmful_prompts = [all_data[i]["prompt"] for i in harmful_indices]
    benign_prompts = [all_data[i]["prompt"] for i in benign_indices]
    pilot_harmful = [all_data[i]["prompt"] for i in harmful_indices[:PILOT_N]]

    # --------------------------------------------------------
    # STEP 1: Baseline predictions (keyword-based)
    # --------------------------------------------------------
    logger.info("Computing baseline predictions...")
    baseline_preds = {r["id"]: baseline_predict(r["prompt"]) for r in all_data}
    baseline_correct = sum(1 for r in all_data if baseline_preds[r["id"]] == r["label"])
    baseline_acc = baseline_correct / n_total
    logger.info(f"Baseline accuracy: {baseline_acc:.3f} ({baseline_correct}/{n_total})")

    # --------------------------------------------------------
    # STEP 2: Layer selection pilot
    # --------------------------------------------------------
    logger.info(f"Layer selection pilot: layers={PILOT_LAYERS}, n={PILOT_N}")
    pilot_base = extract_multilayer(
        MODEL_IDS["base"], pilot_harmful, PILOT_LAYERS, ACTS_DIR / "pilot_base.npz"
    )
    pilot_inst = extract_multilayer(
        MODEL_IDS["instruct"], pilot_harmful, PILOT_LAYERS, ACTS_DIR / "pilot_inst.npz"
    )

    layer_sc_variances: dict[int, float] = {}
    for li in PILOT_LAYERS:
        W_enc_p, b_enc_p, _, _ = load_sae(li)
        z_base_p = sae_project_pretopk(pilot_base[li], W_enc_p, b_enc_p)
        z_inst_p = sae_project_pretopk(pilot_inst[li], W_enc_p, b_enc_p)
        sc_p = np.abs(z_inst_p - z_base_p).mean(axis=0)
        layer_sc_variances[li] = float(sc_p.var())
        del W_enc_p, b_enc_p, z_base_p, z_inst_p, sc_p
        gc.collect()
        logger.info(f"Layer {li}: SC variance = {layer_sc_variances[li]:.6f}")

    PRIMARY_LAYER = max(layer_sc_variances, key=layer_sc_variances.get)
    SECOND_LAYER = sorted(layer_sc_variances, key=layer_sc_variances.get, reverse=True)[1]
    logger.info(f"Primary layer: {PRIMARY_LAYER}, Second: {SECOND_LAYER}")

    del pilot_base, pilot_inst
    gc.collect()

    # --------------------------------------------------------
    # STEP 3: Full activation extraction (all 1000 prompts, PRIMARY_LAYER)
    # --------------------------------------------------------
    logger.info(f"Full extraction at layer {PRIMARY_LAYER}...")

    # Try abliterated model, fallback if needed
    abl_model_id = MODEL_IDS["abliterated"]
    abl_save_path = ACTS_DIR / f"abliterated_layer{PRIMARY_LAYER}.npy"
    if not abl_save_path.exists():
        try:
            extract_single_layer(abl_model_id, all_prompts, PRIMARY_LAYER, abl_save_path)
        except Exception as e:
            logger.error(f"Failed with {abl_model_id}: {e}")
            logger.info(f"Falling back to: {ABLITERATED_FALLBACK}")
            abl_model_id = ABLITERATED_FALLBACK
            extract_single_layer(abl_model_id, all_prompts, PRIMARY_LAYER, abl_save_path)

    for model_key, model_id in [("base", MODEL_IDS["base"]), ("instruct", MODEL_IDS["instruct"])]:
        save_path = ACTS_DIR / f"{model_key}_layer{PRIMARY_LAYER}.npy"
        if not save_path.exists():
            extract_single_layer(model_id, all_prompts, PRIMARY_LAYER, save_path)

    # Load full activations
    acts_all: dict[str, np.ndarray] = {}
    for k in ["base", "instruct", "abliterated"]:
        acts_all[k] = np.load(ACTS_DIR / f"{k}_layer{PRIMARY_LAYER}.npy")  # (n_total, 4096)

    # Slice by label
    acts: dict[str, dict[str, np.ndarray]] = {}
    for k in ["base", "instruct", "abliterated"]:
        acts[k] = {
            "harmful": acts_all[k][harmful_indices],   # (n_harmful, 4096)
            "benign":  acts_all[k][benign_indices],    # (n_benign, 4096)
        }

    # --------------------------------------------------------
    # STEP 4: SAE projection + reconstruction quality gate
    # --------------------------------------------------------
    logger.info(f"Loading SAE for layer {PRIMARY_LAYER}...")
    W_enc, b_enc, W_dec, b_dec = load_sae(PRIMARY_LAYER)

    r2_values: dict[str, float] = {}
    for k in ["base", "instruct", "abliterated"]:
        r2 = compute_r2(acts[k]["harmful"], W_enc, b_enc, W_dec, b_dec)
        r2_values[k] = r2
        logger.info(f"R² {k}: {r2:.4f}")

    USE_PRE_TOPK = (r2_values["instruct"] < 0.85) or (r2_values["abliterated"] < 0.85)
    logger.info(f"Using pre-topk activations: {USE_PRE_TOPK} (R² inst={r2_values['instruct']:.3f}, abl={r2_values['abliterated']:.3f})")

    # Project all prompts through SAE for all models
    logger.info("Projecting through SAE...")
    sae_features: dict[str, dict[str, np.ndarray]] = {}
    for k in ["base", "instruct", "abliterated"]:
        sae_features[k] = {}
        for split in ["harmful", "benign"]:
            z_pre = sae_project_pretopk(acts[k][split], W_enc, b_enc)
            sae_features[k][split] = z_pre if USE_PRE_TOPK else sae_project_posttopk(z_pre)
            logger.debug(f"SAE features {k}/{split}: shape={sae_features[k][split].shape}")

    # Also project all prompts (for per-prompt scoring)
    all_inst_z = sae_project_pretopk(acts_all["instruct"], W_enc, b_enc)   # (n_total, 65536)
    all_base_z = sae_project_pretopk(acts_all["base"], W_enc, b_enc)        # (n_total, 65536)
    all_abl_z  = sae_project_pretopk(acts_all["abliterated"], W_enc, b_enc) # (n_total, 65536)

    # --------------------------------------------------------
    # STEP 5: Safety_change + density filter
    # --------------------------------------------------------
    logger.info("Computing Safety_change per feature...")
    delta_inst = sae_features["instruct"]["harmful"] - sae_features["base"]["harmful"]  # (n_harmful, 65536)
    delta_abl  = sae_features["abliterated"]["harmful"] - sae_features["base"]["harmful"]  # (n_harmful, 65536)

    sc = np.abs(delta_inst).mean(axis=0)  # (65536,) mean Safety_change per feature
    top500_idx = np.argsort(sc)[-500:]   # global indices of top-500

    DENSITY_THRESHOLD = max(1, int(0.10 * n_harmful))
    nonzero_inst = (np.abs(delta_inst[:, top500_idx]) > 1e-6).sum(axis=0)
    nonzero_abl  = (np.abs(delta_abl[:, top500_idx]) > 1e-6).sum(axis=0)
    density_mask = (nonzero_inst >= DENSITY_THRESHOLD) & (nonzero_abl >= DENSITY_THRESHOLD)
    filtered_local = np.where(density_mask)[0]
    filtered_global = top500_idx[filtered_local]
    fraction_surviving = float(len(filtered_global) / 500)
    logger.info(f"Top-500 after density filter: {len(filtered_global)} ({fraction_surviving:.2%})")

    if len(filtered_global) < 50:
        logger.warning("Too few survived at 10% threshold; relaxing to 5%")
        DENSITY_THRESHOLD = max(1, int(0.05 * n_harmful))
        density_mask = (nonzero_inst >= DENSITY_THRESHOLD) & (nonzero_abl >= DENSITY_THRESHOLD)
        filtered_local = np.where(density_mask)[0]
        filtered_global = top500_idx[filtered_local]
        fraction_surviving = float(len(filtered_global) / 500)
        logger.info(f"After relaxation: {len(filtered_global)} ({fraction_surviving:.2%})")

    # --------------------------------------------------------
    # STEP 6: Reversal Score (Pearson correlation per feature)
    # --------------------------------------------------------
    logger.info(f"Computing Reversal Scores for {len(filtered_global)} features...")
    r_values = []
    spearman_values = []
    for fi in filtered_global:
        di = delta_inst[:, fi]
        da = delta_abl[:, fi]
        try:
            r_p, _ = stats.pearsonr(di, da)
            r_s, _ = stats.spearmanr(di, da)
        except Exception:
            r_p = r_s = float("nan")
        r_values.append(float(r_p))
        spearman_values.append(float(r_s))

    r_values = np.array(r_values, dtype=np.float32)
    spearman_values = np.array(spearman_values, dtype=np.float32)

    # Drop NaN entries
    valid_mask = np.isfinite(r_values)
    if not valid_mask.all():
        logger.warning(f"Dropping {(~valid_mask).sum()} NaN Reversal Scores")
        r_values = r_values[valid_mask]
        spearman_values = spearman_values[valid_mask]
        filtered_global = filtered_global[valid_mask]
        filtered_local = filtered_local[valid_mask]

    logger.info(
        f"Reversal Score distribution: mean={r_values.mean():.3f}, "
        f"std={r_values.std():.3f}, min={r_values.min():.3f}, max={r_values.max():.3f}"
    )

    # Hartigan's dip test for bimodality
    try:
        import diptest
        dip_stat, dip_p = diptest.diptest(r_values)
        logger.info(f"Hartigan dip test: stat={dip_stat:.4f}, p={dip_p:.4f}")
    except Exception as e:
        logger.warning(f"diptest failed ({e}), using KDE mode-count fallback")
        from scipy.stats import gaussian_kde
        from scipy.signal import argrelmax
        kde = gaussian_kde(r_values)
        x_grid = np.linspace(float(r_values.min()), float(r_values.max()), 200)
        density = kde(x_grid)
        peaks = argrelmax(density, order=5)[0]
        n_modes = len(peaks)
        dip_stat = float(n_modes)
        dip_p = float(0.01 if n_modes >= 2 else 0.5)
        logger.info(f"KDE fallback: {n_modes} modes, heuristic p={dip_p:.4f}")

    # Empirical Type-I threshold = Q3
    empirical_threshold = float(np.percentile(r_values, 75))
    n_type_i = int((r_values > empirical_threshold).sum())
    n_type_r = int((r_values < 0.0).sum())
    logger.info(f"Q3 threshold={empirical_threshold:.4f}: Type-I={n_type_i}, Type-R={n_type_r}")

    type_i_local = np.where(r_values > empirical_threshold)[0]
    type_r_local = np.where(r_values < 0.0)[0]
    type_i_global = filtered_global[type_i_local]
    type_r_global = filtered_global[type_r_local]

    # Robustness at multiple thresholds
    robustness = {
        f"{T:.2f}": {
            "n_typeI": int((r_values > T).sum()),
            "n_typeR": int((r_values < 0.0).sum()),
        }
        for T in [0.3, 0.5, 0.7, empirical_threshold]
    }

    # --------------------------------------------------------
    # STEP 7: Centroid distances (Criterion 2)
    # --------------------------------------------------------
    logger.info("Computing centroid distances...")
    centroid_results: dict[str, dict] = {}
    for split in ["harmful", "benign"]:
        for type_label, feat_idx in [("typeI", type_i_global), ("typeR", type_r_global)]:
            key = f"{type_label}_{split}"
            if len(feat_idx) == 0:
                centroid_results[key] = {
                    "d_abl_inst_cosine": None, "d_abl_base_cosine": None,
                    "d_inst_base_cosine": None, "d_abl_inst_l2": None, "d_abl_base_l2": None,
                }
            else:
                centroid_results[key] = centroid_distances(
                    inst_sub=sae_features["instruct"][split][:, feat_idx],
                    abl_sub=sae_features["abliterated"][split][:, feat_idx],
                    base_sub=sae_features["base"][split][:, feat_idx],
                )

    criterion2_confirmed = False
    c_ti_h = centroid_results.get("typeI_harmful", {})
    if (c_ti_h.get("d_abl_inst_cosine") is not None
            and c_ti_h.get("d_abl_base_cosine") is not None):
        criterion2_confirmed = bool(
            c_ti_h["d_abl_inst_cosine"] < c_ti_h["d_abl_base_cosine"]
        )
    logger.info(f"Criterion 2 (d_abl_inst < d_abl_base in Type-I/harmful): {criterion2_confirmed}")

    # --------------------------------------------------------
    # STEP 8: Semantic labeling via OpenRouter (Criterion 3)
    # --------------------------------------------------------
    logger.info("Semantic labeling of top-100 salient features...")
    saliency = np.abs(r_values - empirical_threshold) * sc[filtered_global]
    top100_local = np.argsort(saliency)[-100:][::-1]
    top100_global = filtered_global[top100_local]
    top100_r = r_values[top100_local]

    LABELER_PROMPT = (
        "You are analyzing Sparse Autoencoder (SAE) features from a language model.\n\n"
        "Below are text prompts that most strongly activate this SAE feature:\n{segments}\n\n"
        "Classify this feature as EXACTLY ONE of:\n"
        "- harm_concept: Feature encodes understanding of harmful/dangerous/illegal content\n"
        "- safety_refusal: Feature encodes safety mechanisms, refusal patterns, or harm-avoidance\n"
        "- general_capability: Feature encodes general language/reasoning unrelated to harm\n"
        "- other: Ambiguous or unclear\n\n"
        "Respond ONLY with:\nLABEL: <label>\nREASON: <one sentence>"
    )

    inst_harmful_z = sae_features["instruct"]["harmful"]  # (n_harmful, 65536)

    LLM_LABELS: dict[int, dict] = {}
    cumulative_cost = 0.0

    if OPENROUTER_API_KEY:
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        for i, (gidx, r_val) in enumerate(zip(top100_global, top100_r)):
            try:
                feat_acts = inst_harmful_z[:, gidx]
                top5_idx = np.argsort(feat_acts)[-5:][::-1]
                segs = "\n".join(
                    f'  [{j+1}] "{harmful_prompts[pidx][:300].replace(chr(10), " ")}"'
                    for j, pidx in enumerate(top5_idx)
                )
                resp = client.chat.completions.create(
                    model=LABELER_MODEL,
                    messages=[{"role": "user", "content": LABELER_PROMPT.format(segments=segs)}],
                    max_tokens=100,
                    temperature=0.1,
                )
                raw = resp.choices[0].message.content.strip()
                label = "other"
                for line in raw.split("\n"):
                    if line.startswith("LABEL:"):
                        candidate = line.split(":", 1)[1].strip().lower()
                        if candidate in {"harm_concept", "safety_refusal", "general_capability", "other"}:
                            label = candidate
                            break
                LLM_LABELS[int(gidx)] = {"label": label, "r_value": float(r_val), "raw": raw[:300]}
                cumulative_cost += (700 / 1_000_000) * 0.06
                if (i + 1) % 10 == 0:
                    logger.info(f"Labeled {i+1}/100, est. cost ~${cumulative_cost:.3f}")
                if cumulative_cost > 8.0:
                    logger.warning("Cost limit ($8) reached, stopping labeling")
                    break
                time.sleep(0.15)
            except Exception as e:
                logger.error(f"Labeling error for feature {gidx}: {e}")
                LLM_LABELS[int(gidx)] = {"label": "error", "r_value": float(r_val), "raw": str(e)[:200]}
    else:
        # Keyword-based fallback labeler
        logger.warning("No OPENROUTER_API_KEY — using keyword fallback labeler")
        HARM_LBL_KW = ["weapon", "kill", "bomb", "drug", "fraud", "abuse",
                        "illegal", "violence", "terror", "suicide", "hack", "scam"]
        for i, (gidx, r_val) in enumerate(zip(top100_global, top100_r)):
            feat_acts = inst_harmful_z[:, gidx]
            top5_idx = np.argsort(feat_acts)[-5:][::-1]
            text = " ".join(harmful_prompts[pidx][:300].lower() for pidx in top5_idx)
            harm_hits = sum(1 for kw in HARM_LBL_KW if kw in text)
            label = "harm_concept" if harm_hits >= 2 else "general_capability"
            LLM_LABELS[int(gidx)] = {
                "label": label, "r_value": float(r_val),
                "raw": f"keyword_score={harm_hits}",
            }

    # Fisher's exact test: harm_concept rate in Type-I (high-R) vs Type-R (low-R)
    # Fallback: if no Type-R features (all R>0), compare Q3+ vs Q1- groups
    labeled = [
        (int(gidx), float(r), LLM_LABELS[int(gidx)]["label"])
        for gidx, r in zip(top100_global, top100_r)
        if int(gidx) in LLM_LABELS
    ]
    q1_threshold = float(np.percentile(r_values, 25))
    hi_harm  = sum(1 for _, r, lbl in labeled if r > empirical_threshold and lbl == "harm_concept")
    hi_other = sum(1 for _, r, lbl in labeled if r > empirical_threshold and lbl != "harm_concept")
    lo_harm  = sum(1 for _, r, lbl in labeled if r < 0.0 and lbl == "harm_concept")
    lo_other = sum(1 for _, r, lbl in labeled if r < 0.0 and lbl != "harm_concept")
    fisher_fallback_used = False

    if (hi_harm + hi_other > 0) and (lo_harm + lo_other > 0):
        _, fisher_p = fisher_exact([[hi_harm, hi_other], [lo_harm, lo_other]], alternative="greater")
        harm_rate_typeI = float(hi_harm / (hi_harm + hi_other))
        harm_rate_typeR = float(lo_harm / (lo_harm + lo_other))
    else:
        # Fallback: high-R (>Q3) vs low-R (<Q1) when no Type-R features exist
        logger.warning(f"No Type-R features in labeled set; falling back to Q3+/Q1- comparison (Q1={q1_threshold:.3f})")
        fisher_fallback_used = True
        lo_harm  = sum(1 for _, r, lbl in labeled if r < q1_threshold and lbl == "harm_concept")
        lo_other = sum(1 for _, r, lbl in labeled if r < q1_threshold and lbl != "harm_concept")
        if (hi_harm + hi_other > 0) and (lo_harm + lo_other > 0):
            _, fisher_p = fisher_exact([[hi_harm, hi_other], [lo_harm, lo_other]], alternative="greater")
            harm_rate_typeI = float(hi_harm / (hi_harm + hi_other))
            harm_rate_typeR = float(lo_harm / (lo_harm + lo_other))
        else:
            fisher_p = float("nan")
            harm_rate_typeI = float("nan")
            harm_rate_typeR = float("nan")

    logger.info(f"Fisher p={fisher_p}, harm_rate TypeI={harm_rate_typeI}, TypeR/LowR={harm_rate_typeR} (fallback={fisher_fallback_used})")

    # --------------------------------------------------------
    # STEP 9: Criterion evaluation
    # --------------------------------------------------------
    criterion1_confirmed = bool(
        dip_p < 0.05
        or (
            not (isinstance(harm_rate_typeI, float) and np.isnan(harm_rate_typeI))
            and not (isinstance(harm_rate_typeR, float) and np.isnan(harm_rate_typeR))
            and harm_rate_typeR > 0
            and harm_rate_typeI >= 2.0 * harm_rate_typeR
        )
    )
    criterion3_confirmed = bool(
        not (isinstance(fisher_p, float) and np.isnan(fisher_p))
        and fisher_p < 0.05
        and not (isinstance(harm_rate_typeI, float) and np.isnan(harm_rate_typeI))
        and harm_rate_typeI >= 2.0 * harm_rate_typeR
    )
    logger.info(f"Criteria: C1={criterion1_confirmed}, C2={criterion2_confirmed}, C3={criterion3_confirmed}")

    # --------------------------------------------------------
    # STEP 10: Per-prompt SAE harm scoring
    # --------------------------------------------------------
    logger.info("Computing per-prompt SAE harm scores...")

    # Type-I activation: mean activation of Type-I features under instruct model
    if len(type_i_global) > 0:
        all_type_i_act = all_inst_z[:, type_i_global].mean(axis=1)
    else:
        all_type_i_act = np.zeros(n_total, dtype=np.float32)

    # Type-R activation: mean activation of Type-R features under instruct model
    if len(type_r_global) > 0:
        all_type_r_act = all_inst_z[:, type_r_global].mean(axis=1)
    else:
        all_type_r_act = np.zeros(n_total, dtype=np.float32)

    # Safety change score: mean |delta| across top-500 features
    all_delta = np.abs(all_inst_z - all_base_z)
    all_sc_score = all_delta[:, top500_idx].mean(axis=1)
    del all_delta

    # Composite harm score = Type-I activation - Type-R activation
    all_harm_score = all_type_i_act - all_type_r_act

    # Label-free threshold: median of all 1000 prompts (predicts balanced 50/50)
    harm_score_threshold = float(np.median(all_harm_score))

    # Also compute abliteration disruption score per prompt
    # High abl disruption on harmful features = abliteration successfully neutralized harm features
    if len(type_i_global) > 0:
        all_abl_disruption = np.abs(
            all_abl_z[:, type_i_global] - all_base_z[:, type_i_global]
        ).mean(axis=1)
    else:
        all_abl_disruption = np.zeros(n_total, dtype=np.float32)

    del all_inst_z, all_base_z, all_abl_z
    gc.collect()

    sae_preds: dict[str, dict] = {}
    for i, row in enumerate(all_data):
        sae_preds[row["id"]] = {
            "predict_our_method": "harmful" if all_harm_score[i] >= harm_score_threshold else "benign",
            "metadata_sae_harm_score": round(float(all_harm_score[i]), 6),
            "metadata_type_i_activation": round(float(all_type_i_act[i]), 6),
            "metadata_type_r_activation": round(float(all_type_r_act[i]), 6),
            "metadata_sc_score": round(float(all_sc_score[i]), 6),
            "metadata_abl_disruption_score": round(float(all_abl_disruption[i]), 6),
        }

    our_correct = sum(
        1 for row in all_data if sae_preds[row["id"]]["predict_our_method"] == row["label"]
    )
    our_acc = our_correct / n_total
    logger.info(f"Our method accuracy: {our_acc:.3f} ({our_correct}/{n_total})")

    # Detailed per-class accuracy
    our_tp = sum(
        1 for row in all_data
        if row["label"] == "harmful" and sae_preds[row["id"]]["predict_our_method"] == "harmful"
    )
    our_tn = sum(
        1 for row in all_data
        if row["label"] == "benign" and sae_preds[row["id"]]["predict_our_method"] == "benign"
    )
    logger.info(f"Our method TP={our_tp}/{n_harmful}, TN={our_tn}/{n_benign}")

    baseline_tp = sum(
        1 for row in all_data
        if row["label"] == "harmful" and baseline_preds[row["id"]] == "harmful"
    )
    baseline_tn = sum(
        1 for row in all_data
        if row["label"] == "benign" and baseline_preds[row["id"]] == "benign"
    )
    logger.info(f"Baseline TP={baseline_tp}/{n_harmful}, TN={baseline_tn}/{n_benign}")

    # --------------------------------------------------------
    # STEP 11: Format output per exp_gen_sol_out schema
    # --------------------------------------------------------
    logger.info("Formatting output per exp_gen_sol_out schema...")
    examples_by_source: dict[str, list] = defaultdict(list)

    for row in all_data:
        pred = sae_preds[row["id"]]
        example = {
            "input": row["prompt"],
            "output": row["label"],
            "predict_our_method": pred["predict_our_method"],
            "predict_baseline": baseline_preds[row["id"]],
            "metadata_id": row["id"],
            "metadata_category": row.get("category", ""),
            "metadata_source": row.get("source", ""),
            "metadata_char_len": row.get("char_len", 0),
            "metadata_sae_harm_score": pred["metadata_sae_harm_score"],
            "metadata_type_i_activation": pred["metadata_type_i_activation"],
            "metadata_type_r_activation": pred["metadata_type_r_activation"],
            "metadata_sc_score": pred["metadata_sc_score"],
            "metadata_abl_disruption_score": pred["metadata_abl_disruption_score"],
        }
        examples_by_source[row["source"]].append(example)

    datasets = [
        {"dataset": src, "examples": exs}
        for src, exs in sorted(examples_by_source.items())
    ]

    elapsed = time.time() - t_start

    # Serialize NaN-safe
    def _clean(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    method_out = {
        "metadata": _clean({
            "method_name": "SAE Reversal Score Analysis",
            "description": (
                "Mechanistic interpretability of Qwen3-8B base/instruct/abliterated via frozen "
                "Qwen-Scope TopK SAEs. Computes per-feature Reversal_scores (Pearson correlation "
                "of RLHF vs abliteration change vectors)."
            ),
            "primary_layer": int(PRIMARY_LAYER),
            "second_layer": int(SECOND_LAYER),
            "candidate_layers": PILOT_LAYERS,
            "layer_sc_variances": {str(k): v for k, v in layer_sc_variances.items()},
            "sae_repo": SAE_REPO,
            "k_topk": K_TOPK,
            "use_pre_topk": bool(USE_PRE_TOPK),
            "r2_base": r2_values["base"],
            "r2_instruct": r2_values["instruct"],
            "r2_abliterated": r2_values["abliterated"],
            "n_harmful_prompts": n_harmful,
            "n_benign_prompts": n_benign,
            "n_top500_rlhf_sensitive": 500,
            "density_threshold_used": int(DENSITY_THRESHOLD),
            "n_density_filtered": int(len(filtered_global)),
            "fraction_surviving_filter": fraction_surviving,
            "reversal_score_mean": float(r_values.mean()),
            "reversal_score_std": float(r_values.std()),
            "reversal_score_min": float(r_values.min()),
            "reversal_score_max": float(r_values.max()),
            "spearman_r_mean": float(spearman_values.mean()),
            "empirical_type_i_threshold_q3": float(empirical_threshold),
            "n_type_i": int(n_type_i),
            "n_type_r": int(n_type_r),
            "hartigan_dip_stat": float(dip_stat),
            "hartigan_p": float(dip_p),
            "centroid_distances": centroid_results,
            "labeler_model": LABELER_MODEL,
            "n_features_labeled": len(LLM_LABELS),
            "fisher_p": fisher_p,
            "fisher_fallback_used": bool(fisher_fallback_used),
            "fisher_fallback_q1_threshold": float(q1_threshold),
            "harm_concept_rate_typeI": harm_rate_typeI,
            "harm_concept_rate_typeR": harm_rate_typeR,
            "hi_harm_count": int(hi_harm),
            "hi_other_count": int(hi_other),
            "lo_harm_count": int(lo_harm),
            "lo_other_count": int(lo_other),
            "criterion1_bimodality_confirmed": bool(criterion1_confirmed),
            "criterion2_centroid_confirmed": bool(criterion2_confirmed),
            "criterion3_semantic_confirmed": bool(criterion3_confirmed),
            "harm_score_threshold": harm_score_threshold,
            "our_method_accuracy": float(our_acc),
            "baseline_accuracy": float(baseline_acc),
            "our_method_tp": int(our_tp),
            "our_method_tn": int(our_tn),
            "baseline_tp": int(baseline_tp),
            "baseline_tn": int(baseline_tn),
            "robustness_thresholds": robustness,
            "elapsed_seconds": float(elapsed),
            "r_values_list": r_values.tolist(),
            "spearman_r_values_list": spearman_values.tolist(),
            "filtered_global_idx": filtered_global.tolist(),
            "type_i_global_idx": type_i_global.tolist(),
            "type_r_global_idx": type_r_global.tolist(),
            "llm_labels": {str(k): v for k, v in LLM_LABELS.items()},
        }),
        "datasets": datasets,
    }

    out_path = RESULTS / "method_out.json"
    out_path.write_text(json.dumps(method_out, indent=2, ensure_ascii=False))
    logger.info(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    logger.info(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Validate schema
    try:
        from jsonschema import validate as jschema_validate, ValidationError
        schema_path = Path("/ai-inventor/.claude/skills/aii-json/schemas/exp_gen_sol_out.json")
        schema = json.loads(schema_path.read_text())
        jschema_validate(instance=method_out, schema=schema)
        logger.info("Schema validation: PASSED")
    except Exception as e:
        logger.error(f"Schema validation error: {e}")

    # Print summary
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info(f"  Primary layer: {PRIMARY_LAYER}")
    logger.info(f"  R² (instruct): {r2_values['instruct']:.4f}")
    logger.info(f"  Features analyzed: {len(filtered_global)}")
    logger.info(f"  Type-I (R > {empirical_threshold:.3f}): {n_type_i}")
    logger.info(f"  Type-R (R < 0): {n_type_r}")
    logger.info(f"  Hartigan p: {dip_p:.4f}")
    logger.info(f"  Fisher p: {fisher_p}")
    logger.info(f"  Criterion 1 (bimodal): {criterion1_confirmed}")
    logger.info(f"  Criterion 2 (centroid): {criterion2_confirmed}")
    logger.info(f"  Criterion 3 (semantic): {criterion3_confirmed}")
    logger.info(f"  Our method accuracy: {our_acc:.3f}")
    logger.info(f"  Baseline accuracy: {baseline_acc:.3f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
