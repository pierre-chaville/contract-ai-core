from __future__ import annotations

"""CLI: Generate amended-and-restated contracts and instruction JSONs for a dataset.

Usage:
  python dataset/run_reviser.py --template ISDA --model gpt-4.1-mini
"""

import argparse
import json
import logging
from pathlib import Path

from utilities import load_template

from contract_ai_core import (
    ContractReviser,
    ContractReviserConfig,
    ContractTypeTemplate,
    Paragraph,
    split_text_into_paragraphs,
)


def read_text_best_effort(path: Path) -> str:
    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin-1",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
    ]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Generate amended and restated contract")
    parser.add_argument("--template", required=True, help="Template key (e.g., ISDA)")
    parser.add_argument(
        "--model", required=True, help="Model name for reviser (e.g., gpt-4.1-mini)"
    )
    args = parser.parse_args()

    template_key = args.template
    model_name = args.model

    repo_root = Path(__file__).resolve().parents[1]

    contracts_dir = repo_root / "dataset" / "documents" / "amendments" / "initial"
    amendments_dir = repo_root / "dataset" / "documents" / "amendments" / "amendment"
    out_restated_dir = repo_root / "dataset" / "output" / "amendments" / "restated" / model_name
    out_instructions_dir = (
        repo_root / "dataset" / "output" / "amendments" / "instructions" / model_name
    )
    out_restated_dir.mkdir(parents=True, exist_ok=True)
    out_instructions_dir.mkdir(parents=True, exist_ok=True)

    template = ContractTypeTemplate.model_validate(load_template(template_key))

    # Init reviser
    reviser = ContractReviser(ContractReviserConfig(provider="openai", model=model_name))

    # Iterate over amendments; expect matching contract file by stem
    amend_files = sorted(amendments_dir.glob("*.md"))
    if not amend_files:
        logging.warning("No amendment markdown files found in %s", amendments_dir)
        return

    for amend_path in amend_files:
        contract_path = contracts_dir / amend_path.name
        if not contract_path.exists():
            logging.warning("Skipping %s: missing contract %s", amend_path.name, contract_path)
            continue

        logging.info("Processing %s against %s ...", amend_path.name, contract_path.name)

        contract_text = read_text_best_effort(contract_path)
        amendment_text = read_text_best_effort(amend_path)

        contract_paras: list[Paragraph] = split_text_into_paragraphs(contract_text)
        amendment_paras: list[Paragraph] = split_text_into_paragraphs(amendment_text)

        revised = reviser.generate_amended_and_restated(
            contract=contract_paras,
            amendment=amendment_paras,
            template=template,
        )

        # Write restated contract as markdown (paragraphs separated by blank line)
        restated_path = out_restated_dir / amend_path.name
        with restated_path.open("w", encoding="utf-8") as f:
            for p in revised.new_content:
                f.write(p.text)
                f.write("\n\n")

        # Write applied instructions as JSON
        instructions_path = out_instructions_dir / (amend_path.stem + ".json")
        with instructions_path.open("w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "amendment_start_line": r.amendment_start_line,
                        "amendment_end_line": r.amendment_end_line,
                        "amendment_span_text": "\n\n".join(
                            p.text
                            for p in amendment_paras
                            if (
                                r.amendment_start_line is not None
                                and r.amendment_end_line is not None
                                and r.amendment_start_line <= p.index <= r.amendment_end_line
                            )
                        ).strip(),
                        "target_section": r.target_section,
                        "confidence_target": r.confidence_target,
                        "change_explanation": r.change_explanation,
                        "target_paragraph_indices": r.target_paragraph_indices,
                        "confidence_target_paragraph_indices": r.confidence_target_paragraph_indices,
                        "target_paragraph_explanation": r.target_paragraph_explanation,
                        "initial_paragraphs": [
                            {"index": p.index, "text": p.text} for p in (r.initial_paragraphs or [])
                        ],
                        "revised_paragraphs": [
                            {"index": p.index, "text": p.text} for p in (r.revised_paragraphs or [])
                        ],
                        "confidence_revision": r.confidence_revision,
                        "revision_explanation": r.revision_explanation,
                    }
                    for r in revised.applied_instructions
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )

        logging.info("wrote %s", restated_path.relative_to(repo_root))
        logging.info("wrote %s", instructions_path.relative_to(repo_root))


if __name__ == "__main__":
    main()
