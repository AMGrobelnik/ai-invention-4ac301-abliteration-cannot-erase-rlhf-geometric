#!/usr/bin/env python3
"""Convert harmful/benign prompt corpus to exp_sel_data_out schema, grouped by source dataset."""

import json
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger

WORKSPACE = Path(__file__).parent
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(WORKSPACE / "logs" / "data.log", rotation="10 MB", level="DEBUG")

# Source → human-readable dataset name mapping
SOURCE_NAMES = {
    "beavertails":            "beavertails",
    "jailbreakbench":         "jailbreakbench",
    "trustair":               "trustair_jailbreak",
    "harmbench":              "harmbench",
    "advbench":               "advbench",
    "jailbreak_classification": "jailbreak_classification",
    "dolly_15k":              "dolly_15k",
    "toxic_chat":             "toxic_chat",
    "trustair_regular":       "trustair_benign",
}

# Best 4 datasets for SAE mechanistic interpretability of Qwen3 safety models:
# 1. beavertails       - 176 harmful, 14 harm categories (NeurIPS 2023), most diverse
# 2. jailbreakbench    - 55 harmful + 46 benign paired from same curation (NeurIPS 2024)
# 3. harmbench         - 85 harmful, Center for AI Safety standard benchmark
# 4. dolly_15k         - 106 benign, diverse clean human instructions (CC-BY-SA)
SELECTED_SOURCES = {"beavertails", "jailbreakbench", "harmbench", "dolly_15k"}


def main() -> None:
    corpus_path = WORKSPACE / "results" / "data_out.json"
    out_path = WORKSPACE / "full_data_out.json"

    logger.info(f"Loading corpus from {corpus_path}")
    rows = json.loads(corpus_path.read_text())
    logger.info(f"Loaded {len(rows)} rows")

    # Group by source, keep only selected 4
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["source"] in SELECTED_SOURCES:
            by_source[row["source"]].append(row)

    datasets = []
    for source, items in sorted(by_source.items()):
        dataset_name = SOURCE_NAMES.get(source, source)
        examples = []
        for item in items:
            example = {
                "input": item["prompt"],
                "output": item["label"],
                "metadata_category": item["category"],
                "metadata_source": item["source"],
                "metadata_char_len": item["char_len"],
                "metadata_id": item["id"],
                "metadata_task_type": "classification",
                "metadata_n_classes": 2,
            }
            examples.append(example)

        datasets.append({
            "dataset": dataset_name,
            "examples": examples,
        })
        logger.info(
            f"  {dataset_name}: {len(examples)} examples "
            f"(harmful={sum(1 for e in examples if e['output']=='harmful')}, "
            f"benign={sum(1 for e in examples if e['output']=='benign')})"
        )

    output = {"datasets": datasets}
    total_examples = sum(len(ds["examples"]) for ds in datasets)

    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    logger.info(f"Saved full_data_out.json: {len(datasets)} datasets, {total_examples} total examples")

    # Print summary
    logger.info("=== DATASET SUMMARY ===")
    for ds in datasets:
        n = len(ds["examples"])
        h = sum(1 for e in ds["examples"] if e["output"] == "harmful")
        b = n - h
        logger.info(f"  {ds['dataset']:35s}: {n:4d} examples  ({h} harmful / {b} benign)")


if __name__ == "__main__":
    main()
