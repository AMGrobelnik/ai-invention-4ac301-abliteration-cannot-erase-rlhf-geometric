#!/usr/bin/env python3
"""Build 500 harmful + 500 benign prompt corpus for SAE analysis."""

import gc
import io
import json
import math
import os
import random
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from datasketch import MinHash, MinHashLSH
from datasets import load_dataset
from loguru import logger

WORKSPACE = Path(__file__).parent
LOGS_DIR = WORKSPACE / "logs"
RESULTS_DIR = WORKSPACE / "results"
TEMP_DIR = WORKSPACE / "temp" / "datasets"
LOGS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(LOGS_DIR / "build_corpus.log", rotation="30 MB", level="DEBUG")

# --- Resource limits (cgroup v2: 29 GB container) ---
RAM_BUDGET = 8 * 1024**3  # 8 GB for this script
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

HF_TOKEN = os.environ.get("HF_TOKEN", "")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── helpers ──────────────────────────────────────────────────────────────────

def token_estimate(text: str) -> int:
    """Conservative token estimate: chars // 4 (Qwen3 ~3.5 chars/token)."""
    return len(text) // 4


def make_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    chars = text.lower()
    for i in range(max(1, len(chars) - 4)):
        m.update(chars[i:i+5].encode("utf-8"))
    return m


def dedup_candidates(candidates: list[dict], threshold: float = 0.5, num_perm: int = 128) -> list[dict]:
    logger.info(f"Deduplicating {len(candidates)} candidates (threshold={threshold})")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept = []
    for i, item in enumerate(candidates):
        m = make_minhash(item["prompt"], num_perm)
        results = lsh.query(m)
        if not results:
            lsh.insert(str(i), m)
            kept.append(item)
    logger.info(f"After dedup: {len(kept)} unique prompts")
    return kept


def clean_prompt(text: str) -> str | None:
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    if len(text) < 10:
        return None
    return text


# ── source collectors ────────────────────────────────────────────────────────

def collect_beavertails_harmful(target: int = 350) -> list[dict]:
    """PKU-Alignment/BeaverTails 30k_train → harmful prompts."""
    logger.info("Loading BeaverTails harmful prompts...")
    candidates = []
    try:
        ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_train", streaming=True)
        cat_buckets: dict[str, list] = {}
        for row in ds:
            if not row.get("is_safe", True) is False and row.get("is_safe") is not False:
                continue
            prompt = clean_prompt(row.get("prompt", ""))
            if not prompt:
                continue
            if token_estimate(prompt) > 512:
                continue
            # Extract category from dict
            cat_dict = row.get("category", {})
            if isinstance(cat_dict, dict):
                cats = [k for k, v in cat_dict.items() if v]
                category = cats[0] if cats else "harmful_general"
            else:
                category = str(cat_dict) if cat_dict else "harmful_general"
            category = category.replace(",", "/").strip()
            item = {
                "prompt": prompt,
                "label": "harmful",
                "category": category,
                "source": "beavertails",
            }
            cat_buckets.setdefault(category, []).append(item)
            total = sum(len(v) for v in cat_buckets.values())
            if total >= target * 3:
                break

        # Stratified sample
        n_cats = len(cat_buckets)
        per_cat = max(1, target // max(n_cats, 1))
        for cat, items in cat_buckets.items():
            random.shuffle(items)
            candidates.extend(items[:per_cat])
        random.shuffle(candidates)
        candidates = candidates[:target]
        logger.info(f"BeaverTails: collected {len(candidates)} harmful prompts from {n_cats} categories")
    except Exception:
        logger.exception("BeaverTails collection failed")
    return candidates


def collect_jbb_harmful() -> list[dict]:
    """JailbreakBench/JBB-Behaviors harmful split."""
    logger.info("Loading JBB-Behaviors harmful...")
    candidates = []
    try:
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", streaming=True)
        for row in ds["harmful"]:
            prompt = clean_prompt(row.get("Goal", ""))
            if not prompt:
                continue
            candidates.append({
                "prompt": prompt,
                "label": "harmful",
                "category": row.get("Category", "jailbreak"),
                "source": "jailbreakbench",
            })
        logger.info(f"JBB harmful: {len(candidates)} prompts")
    except Exception:
        logger.exception("JBB collection failed")
    return candidates


def collect_trustair_jailbreaks(target: int = 150) -> list[dict]:
    """TrustAIRLab/in-the-wild-jailbreak-prompts jailbreak configs."""
    logger.info("Loading TrustAIRLab jailbreak prompts...")
    candidates = []
    try:
        for config in ["jailbreak_2023_05_07", "jailbreak_2023_12_25"]:
            ds = load_dataset(
                "TrustAIRLab/in-the-wild-jailbreak-prompts",
                config,
                split="train",
                streaming=True,
            )
            for row in ds:
                prompt = clean_prompt(row.get("prompt", ""))
                if not prompt or token_estimate(prompt) > 512:
                    continue
                candidates.append({
                    "prompt": prompt,
                    "label": "harmful",
                    "category": "jailbreak_in_wild",
                    "source": "trustair",
                })
                if len(candidates) >= target * 2:
                    break
            if len(candidates) >= target * 2:
                break
        random.shuffle(candidates)
        candidates = candidates[:target]
        logger.info(f"TrustAIR jailbreaks: {len(candidates)} prompts")
    except Exception:
        logger.exception("TrustAIR collection failed")
    return candidates


def collect_harmbench_github() -> list[dict]:
    """HarmBench standard behaviors from GitHub CSV."""
    logger.info("Loading HarmBench from GitHub...")
    candidates = []
    urls = [
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_test.csv",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text))
                std = df[df["FunctionalCategory"] == "standard"].copy()
                for _, row in std.iterrows():
                    ctx_raw = row.get("ContextString", "")
                    ctx = "" if pd.isna(ctx_raw) else str(ctx_raw).strip()
                    beh_raw = row.get("Behavior", "")
                    beh = "" if pd.isna(beh_raw) else str(beh_raw).strip()
                    prompt = f"{ctx}\n\n{beh}".strip() if ctx else beh
                    prompt = clean_prompt(prompt)
                    if not prompt or token_estimate(prompt) > 512:
                        continue
                    cat_raw = row.get("SemanticCategory", "harmful_behavior")
                    cat = "harmful_behavior" if pd.isna(cat_raw) else str(cat_raw).strip()
                    candidates.append({
                        "prompt": prompt,
                        "label": "harmful",
                        "category": cat,
                        "source": "harmbench",
                    })
                logger.info(f"HarmBench: {len(candidates)} standard behaviors")
                break
        except Exception:
            logger.exception(f"HarmBench URL failed: {url}")
    return candidates


def collect_advbench_github() -> list[dict]:
    """AdvBench harmful behaviors from GitHub."""
    logger.info("Loading AdvBench from GitHub...")
    candidates = []
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            for _, row in df.iterrows():
                prompt = clean_prompt(str(row.get("goal", "") or ""))
                if not prompt or token_estimate(prompt) > 512:
                    continue
                candidates.append({
                    "prompt": prompt,
                    "label": "harmful",
                    "category": "adversarial_behavior",
                    "source": "advbench",
                })
            logger.info(f"AdvBench: {len(candidates)} prompts")
        else:
            logger.warning(f"AdvBench HTTP {r.status_code}")
    except Exception:
        logger.exception("AdvBench collection failed")
    return candidates


def collect_jbb_benign(target: int = 100) -> list[dict]:
    """JailbreakBench benign split."""
    logger.info("Loading JBB-Behaviors benign...")
    candidates = []
    try:
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", streaming=True)
        for row in ds["benign"]:
            prompt = clean_prompt(row.get("Goal", ""))
            if not prompt:
                continue
            candidates.append({
                "prompt": prompt,
                "label": "benign",
                "category": row.get("Category", "benign_general"),
                "source": "jailbreakbench",
            })
        logger.info(f"JBB benign: {len(candidates)} prompts")
    except Exception:
        logger.exception("JBB benign failed")
    return candidates[:target]


def collect_jailbreak_classification_benign(target: int = 250) -> list[dict]:
    """jackhhao/jailbreak-classification benign prompts."""
    logger.info("Loading jailbreak-classification benign...")
    candidates = []
    try:
        ds = load_dataset("jackhhao/jailbreak-classification", split="train", streaming=True)
        for row in ds:
            if row.get("type") != "benign":
                continue
            prompt = clean_prompt(row.get("prompt", ""))
            if not prompt or token_estimate(prompt) > 512:
                continue
            candidates.append({
                "prompt": prompt,
                "label": "benign",
                "category": "benign_instruction",
                "source": "jailbreak_classification",
            })
            if len(candidates) >= target:
                break
        logger.info(f"JailbreakClassification benign: {len(candidates)} prompts")
    except Exception:
        logger.exception("JailbreakClassification benign failed")
    return candidates


def collect_dolly_benign(target: int = 200) -> list[dict]:
    """databricks/databricks-dolly-15k instruction field as benign prompts."""
    logger.info("Loading Dolly-15k benign...")
    candidates = []
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
        for row in ds:
            prompt = clean_prompt(row.get("instruction", ""))
            if not prompt or token_estimate(prompt) > 512:
                continue
            candidates.append({
                "prompt": prompt,
                "label": "benign",
                "category": row.get("category", "benign_instruction"),
                "source": "dolly_15k",
            })
            if len(candidates) >= target:
                break
        logger.info(f"Dolly-15k benign: {len(candidates)} prompts")
    except Exception:
        logger.exception("Dolly-15k benign failed")
    return candidates


def collect_toxic_chat_benign(target: int = 200) -> list[dict]:
    """lmsys/toxic-chat non-toxic prompts as benign."""
    logger.info("Loading ToxicChat benign (non-toxic)...")
    candidates = []
    try:
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train", streaming=True)
        for row in ds:
            if row.get("toxicity", 1) != 0:
                continue
            prompt = clean_prompt(row.get("user_input", ""))
            if not prompt or token_estimate(prompt) > 512:
                continue
            # Filter non-English (simple heuristic: ASCII ratio)
            ascii_ratio = sum(1 for c in prompt if ord(c) < 128) / len(prompt)
            if ascii_ratio < 0.8:
                continue
            candidates.append({
                "prompt": prompt,
                "label": "benign",
                "category": "real_user_benign",
                "source": "toxic_chat",
            })
            if len(candidates) >= target:
                break
        logger.info(f"ToxicChat benign: {len(candidates)} prompts")
    except Exception:
        logger.exception("ToxicChat benign failed")
    return candidates


def collect_trustair_regular(target: int = 200) -> list[dict]:
    """TrustAIRLab regular (non-jailbreak) prompts as benign."""
    logger.info("Loading TrustAIRLab regular (benign) prompts...")
    candidates = []
    try:
        for config in ["regular_2023_05_07", "regular_2023_12_25"]:
            ds = load_dataset(
                "TrustAIRLab/in-the-wild-jailbreak-prompts",
                config,
                split="train",
                streaming=True,
            )
            for row in ds:
                prompt = clean_prompt(row.get("prompt", ""))
                if not prompt or token_estimate(prompt) > 512:
                    continue
                candidates.append({
                    "prompt": prompt,
                    "label": "benign",
                    "category": "real_user_benign",
                    "source": "trustair_regular",
                })
                if len(candidates) >= target:
                    break
            if len(candidates) >= target:
                break
        logger.info(f"TrustAIR regular: {len(candidates)} prompts")
    except Exception:
        logger.exception("TrustAIR regular failed")
    return candidates


# ── balancing ────────────────────────────────────────────────────────────────

def cap_categories(items: list[dict], max_frac: float = 0.35, total: int = 500) -> list[dict]:
    """Cap any single category to max_frac * total."""
    cap = int(max_frac * total)
    from collections import Counter
    cat_counts: Counter = Counter()
    result = []
    for item in items:
        cat = item["category"]
        if cat_counts[cat] < cap:
            result.append(item)
            cat_counts[cat] += 1
    return result


def final_sample(pool: list[dict], n: int) -> list[dict]:
    random.shuffle(pool)
    return pool[:n]


def assign_ids(items: list[dict], prefix: str) -> list[dict]:
    out = []
    for i, item in enumerate(sorted(items, key=lambda x: (x["source"], x["category"]))):
        item = dict(item)
        item["id"] = f"{prefix}_{i:03d}"
        item["char_len"] = len(item["prompt"])
        out.append(item)
    return out


def validate_row(row: dict) -> bool:
    required = {"id", "prompt", "label", "category", "source", "char_len"}
    if not required.issubset(row.keys()):
        return False
    if row["label"] not in ("harmful", "benign"):
        return False
    if not row["prompt"] or not isinstance(row["prompt"], str):
        return False
    if not isinstance(row["char_len"], int):
        return False
    return True


# ── main ─────────────────────────────────────────────────────────────────────

@logger.catch(reraise=True)
def main():
    logger.info("=== Building harmful/benign prompt corpus ===")

    # --- COLLECT HARMFUL ---
    harmful_pool = []

    bt = collect_beavertails_harmful(target=350)
    harmful_pool.extend(bt)
    logger.info(f"After BeaverTails: {len(harmful_pool)} harmful candidates")

    jbb_h = collect_jbb_harmful()
    harmful_pool.extend(jbb_h)

    trustair = collect_trustair_jailbreaks(target=150)
    harmful_pool.extend(trustair)

    harmbench = collect_harmbench_github()
    harmful_pool.extend(harmbench)

    advbench = collect_advbench_github()
    harmful_pool.extend(advbench)

    logger.info(f"Total harmful candidates before dedup: {len(harmful_pool)}")

    # --- COLLECT BENIGN ---
    benign_pool = []

    benign_pool.extend(collect_jbb_benign(target=100))
    benign_pool.extend(collect_jailbreak_classification_benign(target=250))
    benign_pool.extend(collect_dolly_benign(target=200))
    benign_pool.extend(collect_toxic_chat_benign(target=200))
    benign_pool.extend(collect_trustair_regular(target=200))

    logger.info(f"Total benign candidates before dedup: {len(benign_pool)}")
    gc.collect()

    # --- DEDUP (all together to catch cross-set near-dups) ---
    all_candidates = harmful_pool + benign_pool
    all_deduped = dedup_candidates(all_candidates, threshold=0.5, num_perm=128)

    harmful_deduped = [x for x in all_deduped if x["label"] == "harmful"]
    benign_deduped = [x for x in all_deduped if x["label"] == "benign"]
    logger.info(f"Post-dedup: {len(harmful_deduped)} harmful, {len(benign_deduped)} benign")

    # If short on harmful, try to supplement with remaining pool
    if len(harmful_deduped) < 500:
        logger.warning(f"Harmful pool only {len(harmful_deduped)}, using all available")

    # --- TOKEN FILTER (already done during collection, but double-check) ---
    harmful_deduped = [x for x in harmful_deduped if token_estimate(x["prompt"]) <= 512]
    benign_deduped = [x for x in benign_deduped if token_estimate(x["prompt"]) <= 512]
    logger.info(f"Post-token-filter: {len(harmful_deduped)} harmful, {len(benign_deduped)} benign")

    # --- CATEGORY CAP (no single harmful category > 35%) ---
    harmful_capped = cap_categories(harmful_deduped, max_frac=0.35, total=500)
    logger.info(f"Post-category-cap: {len(harmful_capped)} harmful")

    # --- FINAL SAMPLE to 500 each ---
    n_harmful = min(500, len(harmful_capped))
    n_benign = min(500, len(benign_deduped))
    harmful_final = final_sample(harmful_capped, n_harmful)
    benign_final = final_sample(benign_deduped, n_benign)

    # Pad if needed (relax dedup by pulling from original pool)
    if len(harmful_final) < 500:
        logger.warning(f"Only {len(harmful_final)} harmful prompts available (target 500)")
    if len(benign_final) < 500:
        logger.warning(f"Only {len(benign_final)} benign prompts available (target 500)")

    # --- ASSIGN IDs ---
    harmful_final = assign_ids(harmful_final, "h")
    benign_final = assign_ids(benign_final, "b")

    # --- COMBINE AND SHUFFLE ---
    all_rows = harmful_final + benign_final
    random.seed(RANDOM_SEED)
    random.shuffle(all_rows)

    # --- VALIDATE ---
    valid_rows = [r for r in all_rows if validate_row(r)]
    invalid = len(all_rows) - len(valid_rows)
    if invalid:
        logger.warning(f"{invalid} rows failed validation and were dropped")
    all_rows = valid_rows

    # --- STATISTICS ---
    h_rows = [r for r in all_rows if r["label"] == "harmful"]
    b_rows = [r for r in all_rows if r["label"] == "benign"]

    from collections import Counter
    h_cats = Counter(r["category"] for r in h_rows)
    b_sources = Counter(r["source"] for r in b_rows)
    h_sources = Counter(r["source"] for r in h_rows)

    char_lens = [r["char_len"] for r in all_rows]
    logger.info(f"=== CORPUS STATS ===")
    logger.info(f"Total rows: {len(all_rows)} ({len(h_rows)} harmful, {len(b_rows)} benign)")
    logger.info(f"Char len: min={min(char_lens)}, mean={sum(char_lens)//len(char_lens)}, max={max(char_lens)}")
    logger.info(f"Harmful categories: {dict(h_cats.most_common(10))}")
    logger.info(f"Harmful sources: {dict(h_sources)}")
    logger.info(f"Benign sources: {dict(b_sources)}")

    # --- SAVE data_out.json ---
    out_path = RESULTS_DIR / "data_out.json"
    out_path.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2))
    logger.info(f"Saved data_out.json: {len(all_rows)} rows ({out_path.stat().st_size/1024:.1f} KB)")

    # --- SAVE data_mini.json (100 rows: 50 harmful + 50 benign, category-balanced) ---
    # 50 harmful: ~5 per top category
    h_mini: list[dict] = []
    h_by_cat: dict[str, list] = {}
    for r in h_rows:
        h_by_cat.setdefault(r["category"], []).append(r)
    top_cats = sorted(h_by_cat.keys(), key=lambda c: -len(h_by_cat[c]))
    n_top = min(10, len(top_cats))
    per_cat_mini = max(1, 50 // max(n_top, 1))
    for cat in top_cats[:n_top]:
        h_mini.extend(h_by_cat[cat][:per_cat_mini])
        if len(h_mini) >= 50:
            break
    h_mini = h_mini[:50]

    # 50 benign: proportional by source
    b_mini: list[dict] = []
    total_b = len(b_rows)
    for src, cnt in b_sources.most_common():
        n = max(1, round(cnt / total_b * 50))
        b_of_src = [r for r in b_rows if r["source"] == src]
        b_mini.extend(b_of_src[:n])
        if len(b_mini) >= 50:
            break
    b_mini = b_mini[:50]

    mini_rows = h_mini + b_mini
    random.shuffle(mini_rows)
    mini_path = RESULTS_DIR / "data_mini.json"
    mini_path.write_text(json.dumps(mini_rows, ensure_ascii=False, indent=2))
    logger.info(f"Saved data_mini.json: {len(mini_rows)} rows")

    # --- SAVE data_preview.json (10 rows: 5 harmful + 5 benign) ---
    preview_h = []
    seen_cats = set()
    for r in h_rows:
        if r["category"] not in seen_cats:
            preview_h.append(r)
            seen_cats.add(r["category"])
        if len(preview_h) >= 5:
            break
    if len(preview_h) < 5:
        preview_h = h_rows[:5]

    preview_b = []
    seen_srcs = set()
    for r in b_rows:
        if r["source"] not in seen_srcs:
            preview_b.append(r)
            seen_srcs.add(r["source"])
        if len(preview_b) >= 5:
            break
    if len(preview_b) < 5:
        preview_b = b_rows[:5]

    preview_rows = preview_h[:5] + preview_b[:5]
    preview_path = RESULTS_DIR / "data_preview.json"
    preview_path.write_text(json.dumps(preview_rows, ensure_ascii=False, indent=2))
    logger.info(f"Saved data_preview.json: {len(preview_rows)} rows")

    # --- SAVE metadata ---
    meta = {
        "total": len(all_rows),
        "harmful": len(h_rows),
        "benign": len(b_rows),
        "harmful_categories": dict(h_cats),
        "harmful_sources": dict(h_sources),
        "benign_sources": dict(b_sources),
        "char_len_stats": {
            "min": min(char_lens),
            "mean": sum(char_lens) // len(char_lens),
            "max": max(char_lens),
            "p25": int(np.percentile(char_lens, 25)),
            "p50": int(np.percentile(char_lens, 50)),
            "p75": int(np.percentile(char_lens, 75)),
            "p95": int(np.percentile(char_lens, 95)),
        },
        "schema": {
            "fields": ["id", "prompt", "label", "category", "source", "char_len"],
            "label_values": ["harmful", "benign"],
        },
        "sources": {
            "beavertails": "PKU-Alignment/BeaverTails NeurIPS 2023, 14 harm categories",
            "jailbreakbench": "JailbreakBench/JBB-Behaviors NeurIPS 2024, 100 harmful + 100 benign behaviors",
            "trustair": "TrustAIRLab/in-the-wild-jailbreak-prompts ACM CCS 2024, real-world jailbreaks",
            "harmbench": "HarmBench Center for AI Safety 2024, standard harmful behaviors",
            "advbench": "AdvBench llm-attacks Zou et al. 2023, 520 adversarial behaviors",
            "jailbreak_classification": "jackhhao/jailbreak-classification, 5721 benign prompts",
            "dolly_15k": "databricks/databricks-dolly-15k, CC-BY-SA 3.0, 15K benign instructions",
            "toxic_chat": "lmsys/toxic-chat ToxicChat 2023, real user prompts labeled toxic/non-toxic",
            "trustair_regular": "TrustAIRLab regular (non-jailbreak) prompts as benign",
        },
    }
    meta_path = RESULTS_DIR / "metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    logger.info(f"Saved metadata.json")
    logger.info("=== DONE ===")


if __name__ == "__main__":
    os.chdir(WORKSPACE)
    main()
