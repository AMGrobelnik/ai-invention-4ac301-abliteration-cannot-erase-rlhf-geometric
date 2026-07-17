#!/usr/bin/env python3
"""Smoke test: 6 prompts, 1 layer, skip LLM labeling."""
import gc
import json
import math
import os
import sys
import time
import resource
from pathlib import Path
from typing import Optional

os.environ.setdefault("HF_HOME", "/ai-inventor/aii_data/hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/ai-inventor/aii_data/hf_cache/hub")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

import numpy as np
import scipy.stats as stats
import psutil
from loguru import logger
import torch
from scipy.spatial.distance import cosine

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

WORKSPACE = Path("/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_2/gen_art/gen_art_experiment_1")
ACTS_DIR = WORKSPACE / "activations"
ACTS_DIR.mkdir(exist_ok=True)

DATASET_PATH = Path("/ai-inventor/aii_data/runs/run_hql72-TZXK98/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/results/data_out.json")
SAE_REPO = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
D_MODEL = 4096
N_FEATURES = 65536
K_TOPK = 50
TEST_LAYER = 14  # Just one layer for smoke test
N_SMOKE = 6  # 3 harmful + 3 benign

MODEL_IDS = {
    "base": "Qwen/Qwen3-8B-Base",
    "instruct": "Qwen/Qwen3-8B",
    "abliterated": "huihui-ai/Huihui-Qwen3-8B-abliterated-v2",
}

HAS_GPU = torch.cuda.is_available()
logger.info(f"GPU: {HAS_GPU}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB" if HAS_GPU else "GPU: False")


def load_sae(layer_idx: int):
    from huggingface_hub import hf_hub_download
    sae_file = hf_hub_download(SAE_REPO, f"layer{layer_idx}.sae.pt")
    sae = torch.load(sae_file, map_location="cpu", weights_only=True)
    logger.info(f"SAE keys: {list(sae.keys())}")
    logger.info(f"SAE shapes: {[(k, tuple(v.shape)) for k, v in sae.items()]}")
    W_enc = sae["W_enc"].float().numpy()
    b_enc = sae["b_enc"].float().numpy()
    W_dec = sae["W_dec"].float().numpy()
    b_dec = sae["b_dec"].float().numpy()
    if W_enc.shape == (D_MODEL, N_FEATURES):
        W_enc = W_enc.T
    if W_dec.shape == (N_FEATURES, D_MODEL):
        W_dec = W_dec.T
    assert W_enc.shape == (N_FEATURES, D_MODEL), f"W_enc: {W_enc.shape}"
    assert W_dec.shape == (D_MODEL, N_FEATURES), f"W_dec: {W_dec.shape}"
    del sae; gc.collect()
    return W_enc, b_enc, W_dec, b_dec


def extract_single_layer(model_id: str, prompts: list, layer_idx: int, save_path: Path):
    if save_path.exists():
        logger.info(f"Cache hit: {save_path}")
        return np.load(save_path)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    acts = []
    hook_out = {}
    def _hook(module, input, output):
        # transformers 5.x: Qwen3 decoder returns hidden_states tensor directly
        h = output[0] if isinstance(output, (tuple, list)) else output
        hook_out["x"] = h[:, -1, :].detach().cpu().float().numpy()
    h = model.model.layers[layer_idx].register_forward_hook(_hook)
    with torch.no_grad():
        for i, p in enumerate(prompts):
            inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            model(**inputs)
            acts.append(hook_out["x"][0])
            logger.info(f"  Prompt {i+1}/{len(prompts)}: shape={hook_out['x'].shape}")
    h.remove()
    # Sanity: base and instruct should differ
    if len(acts) >= 2:
        diff = np.linalg.norm(acts[0] - acts[-1])
        logger.info(f"  L2 norm of first vs last activation: {diff:.4f} (non-zero = good)")
    del model; gc.collect(); torch.cuda.empty_cache()
    result = np.stack(acts, axis=0)
    np.save(save_path, result)
    logger.info(f"Saved {save_path}: shape={result.shape}")
    return result


def main():
    t0 = time.time()

    # Load small subset
    data = json.loads(DATASET_PATH.read_text())
    harmful = [r for r in data if r["label"] == "harmful"][:3]
    benign  = [r for r in data if r["label"] == "benign"][:3]
    smoke_data = harmful + benign
    prompts = [r["prompt"] for r in smoke_data]
    logger.info(f"Smoke test: {len(prompts)} prompts")
    logger.info(f"Harmful: {[p[:50] for p in prompts[:3]]}")

    # Extract activations for all 3 models
    acts = {}
    for k, model_id in MODEL_IDS.items():
        save_path = ACTS_DIR / f"smoke_{k}_layer{TEST_LAYER}.npy"
        try:
            acts[k] = extract_single_layer(model_id, prompts, TEST_LAYER, save_path)
        except Exception as e:
            logger.error(f"Failed {model_id}: {e}")
            raise

    # Verify shapes
    for k, arr in acts.items():
        assert arr.shape == (len(prompts), D_MODEL), f"{k}: {arr.shape}"
        logger.info(f"  {k}: shape={arr.shape}, mean={arr.mean():.4f}")

    # Verify models differ
    diff_inst = np.linalg.norm(acts["instruct"] - acts["base"])
    diff_abl  = np.linalg.norm(acts["abliterated"] - acts["base"])
    logger.info(f"L2 diff instruct-base: {diff_inst:.4f}")
    logger.info(f"L2 diff abl-base: {diff_abl:.4f}")
    assert diff_inst > 0, "instruct == base! Hook may be wrong"
    assert diff_abl > 0, "abliterated == base! Hook may be wrong"

    # Load SAE
    logger.info(f"Loading SAE for layer {TEST_LAYER}...")
    W_enc, b_enc, W_dec, b_dec = load_sae(TEST_LAYER)
    logger.info(f"SAE W_enc shape: {W_enc.shape}, W_dec shape: {W_dec.shape}")

    # SAE projection
    z_base = acts["base"] @ W_enc.T + b_enc           # (6, 65536)
    z_inst = acts["instruct"] @ W_enc.T + b_enc       # (6, 65536)
    z_abl  = acts["abliterated"] @ W_enc.T + b_enc    # (6, 65536)
    logger.info(f"z_base: shape={z_base.shape}, min={z_base.min():.4f}, max={z_base.max():.4f}")

    # Verify z_base sparsity (pre-topk)
    # TopK gate test
    def topk(z):
        z_post = np.zeros_like(z)
        for i in range(len(z)):
            idx = np.argpartition(z[i], -K_TOPK)[-K_TOPK:]
            z_post[i, idx] = z[i, idx]
        return z_post
    z_post_base = topk(z_base)
    nonzero_per_sample = (z_post_base > 0).sum(axis=1)
    logger.info(f"Post-topk non-zeros per sample (should be ~{K_TOPK}): {nonzero_per_sample.tolist()}")

    # R² test
    recon = z_post_base @ W_dec.T + b_dec
    a = acts["base"]
    ss_res = ((a - recon) ** 2).sum(axis=1).mean()
    ss_tot = ((a - a.mean(axis=0)) ** 2).sum(axis=1).mean()
    r2_base = 1.0 - ss_res / (ss_tot + 1e-12)
    logger.info(f"R² base: {r2_base:.4f} (expected close to 1.0)")
    # With only 6 samples R² may be low; just check it's finite
    assert np.isfinite(r2_base), "R² is NaN!"

    # Safety change
    harmful_idx = list(range(3))
    delta_inst = z_inst[harmful_idx] - z_base[harmful_idx]  # (3, 65536)
    delta_abl  = z_abl[harmful_idx]  - z_base[harmful_idx]  # (3, 65536)
    sc = np.abs(delta_inst).mean(axis=0)
    logger.info(f"SC: max={sc.max():.4f}, mean={sc.mean():.6f}")
    assert sc.max() > 0, "Safety change is all zero!"

    # Top-5 features for quick reversal score
    top5_idx = np.argsort(sc)[-5:]
    r_vals = []
    for fi in top5_idx:
        di = delta_inst[:, fi]
        da = delta_abl[:, fi]
        try:
            r, _ = stats.pearsonr(di, da)
            r_vals.append(float(r))
        except Exception:
            r_vals.append(float("nan"))
    logger.info(f"Reversal scores (top-5 features): {[f'{v:.3f}' for v in r_vals]}")
    assert any(np.isfinite(r) for r in r_vals), "All reversal scores are NaN!"

    # OpenRouter check (skip if no key)
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        resp = client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": "LABEL: harm_concept\nREASON: test"}],
            max_tokens=10, temperature=0.0,
        )
        logger.info(f"OpenRouter test: {resp.choices[0].message.content[:50]}")
    else:
        logger.warning("No OPENROUTER_API_KEY, skipping OpenRouter test")

    # Test output schema compatibility
    from collections import defaultdict
    examples_by_source = defaultdict(list)
    for row in smoke_data:
        examples_by_source[row["source"]].append({
            "input": row["prompt"],
            "output": row["label"],
            "predict_our_method": "harmful",
            "predict_baseline": "harmful",
            "metadata_id": row["id"],
            "metadata_category": row.get("category", ""),
            "metadata_source": row.get("source", ""),
            "metadata_char_len": row.get("char_len", 0),
            "metadata_sae_harm_score": 0.123,
            "metadata_type_i_activation": 0.456,
            "metadata_type_r_activation": 0.789,
            "metadata_sc_score": 0.012,
            "metadata_abl_disruption_score": 0.034,
        })
    datasets = [{"dataset": src, "examples": exs} for src, exs in examples_by_source.items()]
    smoke_out = {"metadata": {"test": True}, "datasets": datasets}
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(smoke_out, f)
        tmp_path = f.name

    # Validate schema using jsonschema directly
    from jsonschema import validate as jschema_validate, ValidationError
    import json as _json
    schema_path = "/ai-inventor/.claude/skills/aii-json/schemas/exp_gen_sol_out.json"
    schema = _json.loads(Path(schema_path).read_text())
    data_to_validate = _json.loads(Path(tmp_path).read_text())
    try:
        jschema_validate(instance=data_to_validate, schema=schema)
        logger.info("Schema validation: PASSED")
    except ValidationError as e:
        raise AssertionError(f"Schema validation FAILED: {e.message}")

    elapsed = time.time() - t0
    logger.info(f"SMOKE TEST PASSED in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
