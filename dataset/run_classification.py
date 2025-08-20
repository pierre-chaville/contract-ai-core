from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List
from load_template import load_template

from contract_ai_core import (
    ClauseClassifier,
    ClauseClassifierConfig,
    ContractTypeTemplate,
    Paragraph,
    split_text_into_paragraphs,
)


def read_markdown(path: Path) -> str:
    candidate_encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin-1",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
    ]
    for enc in candidate_encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def classify_file(
    classifier: ClauseClassifier,
    template: ContractTypeTemplate,
    md_path: Path,
) -> dict:
    text = read_markdown(md_path)
    paragraphs: List[Paragraph] = split_text_into_paragraphs(text)
    classification = classifier.classify_paragraphs(paragraphs, template, source_id=md_path.name)

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
    temperature = 1.0 if 'gpt-5' in model_name else 0.0

    classifier = ClauseClassifier(
        ClauseClassifierConfig(provider="openai", model=model_name, temperature=temperature)
    )
    template = ContractTypeTemplate.model_validate(load_template(template_key))

    md_files = sorted(input_dir.glob("*.md"))
    if not md_files:
        print(f"No markdown files found in {input_dir}")
        return

    for md_path in md_files:
        print(f"Classifying {md_path.name} ...")
        result = classify_file(classifier, template, md_path)

        out_path = output_dir / (md_path.stem + ".csv")
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "clause_key", "confidence", "text"])
            writer.writerows(result["rows"])  # type: ignore[index]
        print(f"  -> wrote {out_path}")


if __name__ == "__main__":
    main()
