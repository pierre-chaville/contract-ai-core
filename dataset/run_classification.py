from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List
from utilities import load_template, write_tokens_usage

from contract_ai_core import (
    ClauseClassifier,
    ClauseClassifierConfig,
    ContractTypeTemplate,
    Paragraph,
    split_text_into_paragraphs,
)
from datetime import datetime, timezone


def classify_file(
    classifier: ClauseClassifier,
    template: ContractTypeTemplate,
    md_path: Path,
    model_name: str,
    repo_root: Path,
) -> dict:
    text = md_path.read_text(encoding="utf-8")
    paragraphs: List[Paragraph] = split_text_into_paragraphs(text)
    classification, usage = classifier.classify_paragraphs_with_usage(
        paragraphs, template, source_id=md_path.name
    )
    write_tokens_usage("clauses", md_path.name, model_name, usage, len(paragraphs), repo_root)

    rows = []
    for cp in classification.paragraphs:
        idx = cp.paragraph.index
        key = cp.clause_key or ""
        conf_pct = round(cp.confidence * 100) if cp.confidence is not None else ""
        txt = cp.paragraph.text
        rows.append([idx, key, conf_pct, txt])
    return {"rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify contract clauses for a template")
    parser.add_argument("--template", required=True, help="Template key (e.g., ISDA)")
    parser.add_argument("--model", required=True, help="Model name (e.g., gpt-4.1)")
    args = parser.parse_args()

    template_key = args.template
    model_name = args.model

    repo_root = Path(__file__).resolve().parents[1]

    input_dir = repo_root / "dataset" / "documents" / template_key
    output_dir = repo_root / "dataset" / "output" / "clauses" / template_key / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # temperature for gpt-5 is different from other models
    temperature = 1.0 if "gpt-5" in model_name else 0.0

    classifier = ClauseClassifier(
        ClauseClassifierConfig(provider="openai", model=model_name, temperature=temperature)
    )
    template = ContractTypeTemplate.model_validate(load_template(template_key))

    md_files = sorted(input_dir.glob("*.md"))
    if not md_files:
        print(f"No markdown files found in {input_dir}")
        return

    for md_path in md_files:
        if os.path.exists(output_dir / (md_path.stem + ".csv")):
            print(f"Skipping {md_path.name} because it already exists")
            continue
        print(f"Classifying {md_path.name} ...")
        result = classify_file(classifier, template, md_path, model_name, repo_root)

        out_path = output_dir / (md_path.stem + ".csv")
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "clause_key", "confidence", "text"])
            writer.writerows(result["rows"])  # type: ignore[index]


if __name__ == "__main__":
    main()
