from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from contract_ai_core import (
    ClauseClassifier,
    ClauseClassifierConfig,
    ContractTypeTemplate,
    Paragraph,
    text_to_paragraphs,
)
from utilities import load_template, write_tokens_usage


def classify_file(
    classifier: ClauseClassifier,
    template: ContractTypeTemplate,
    txt_path: Path,
    model_name: str,
    repo_root: Path,
) -> dict:
    text = txt_path.read_text(encoding="utf-8")
    paragraphs: list[Paragraph] = text_to_paragraphs(text)
    classification, usage = classifier.classify_paragraphs_with_usage(
        paragraphs, template, source_id=txt_path.name
    )
    write_tokens_usage("clauses", txt_path.name, model_name, usage, len(paragraphs), repo_root)

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
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "azure", "anthropic"],
        help="LLM provider (default: openai)",
    )
    args = parser.parse_args()

    template_key = args.template
    model_name = args.model
    provider = args.provider

    repo_root = Path(__file__).resolve().parents[1]

    input_dir = repo_root / "dataset" / "documents" / "contracts" / template_key
    output_dir = repo_root / "dataset" / "output" / "clauses" / template_key / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # temperature for gpt-5 is different from other models
    temperature = 1.0 if "gpt-5" in model_name else 0.0

    classifier = ClauseClassifier(
        ClauseClassifierConfig(
            provider=provider, model=model_name, temperature=temperature, max_tokens=8000
        )
    )
    template = ContractTypeTemplate.model_validate(load_template(template_key))

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"No text files found in {input_dir}")
        return

    for txt_path in txt_files:
        if os.path.exists(output_dir / (txt_path.stem + ".csv")):
            print(f"Skipping {txt_path.name} because it already exists")
            continue
        print(f"Classifying {txt_path.name} using {provider}:{model_name} ...")
        result = classify_file(classifier, template, txt_path, model_name, repo_root)

        out_path = output_dir / (txt_path.stem + ".csv")
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "clause_key", "confidence", "text"])
            writer.writerows(result["rows"])  # type: ignore[index]


if __name__ == "__main__":
    main()
