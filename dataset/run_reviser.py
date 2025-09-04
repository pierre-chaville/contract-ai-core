from __future__ import annotations

"""CLI: Generate amended-and-restated contracts and instruction JSONs for a dataset.

Usage:
  python dataset/run_reviser.py --template ISDA --model gpt-4.1-mini
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from contract_ai_core import (
    ContractReviser,
    ContractReviserConfig,
    ContractTypeTemplate,
    Paragraph,
    split_text_into_paragraphs,
)
from utilities import load_template


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
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "azure", "anthropic"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--step",
        default="all",
        choices=["all", "1", "2", "3"],
        help="Step to run (default: all)",
    )
    args = parser.parse_args()

    template_key = args.template
    model_name = args.model
    provider = args.provider
    step = args.step

    repo_root = Path(__file__).resolve().parents[1]

    template = ContractTypeTemplate.model_validate(load_template(template_key))

    # Init reviser
    reviser = ContractReviser(
        ContractReviserConfig(provider=provider, model=model_name, max_tokens=8000)
    )

    if step == "all":
        run_step_all(repo_root, model_name, template, reviser)
    elif step == "1":
        run_step_1(repo_root, model_name, template, reviser)
    elif step == "2":
        run_step_2(repo_root, model_name, template, reviser)
    elif step == "3":
        run_step_3(repo_root, model_name, template, reviser)
    else:
        raise ValueError(f"Invalid step: {step}")


def run_step_all(
    repo_root: Path, model_name: str, template: ContractTypeTemplate, reviser: ContractReviser
) -> None:
    contracts_dir = repo_root / "dataset" / "documents" / "amendments" / "initial"
    amendments_dir = repo_root / "dataset" / "documents" / "amendments" / "amendment"
    out_restated_dir = repo_root / "dataset" / "output" / "amendments" / "restated" / model_name
    out_instructions_dir = (
        repo_root / "dataset" / "output" / "amendments" / "instructions" / model_name
    )
    out_restated_dir.mkdir(parents=True, exist_ok=True)
    out_instructions_dir.mkdir(parents=True, exist_ok=True)

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


def run_step_1(
    repo_root: Path, model_name: str, template: ContractTypeTemplate, reviser: ContractReviser
) -> None:
    amendments_dir = repo_root / "dataset" / "documents" / "amendments" / "amendment"
    out_elementary_dir = repo_root / "dataset" / "output" / "amendments" / "elementary" / model_name
    out_elementary_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over amendments; expect matching contract file by stem
    amend_files = sorted(amendments_dir.glob("*.md"))
    if not amend_files:
        logging.warning("No amendment markdown files found in %s", amendments_dir)
        return

    for amend_path in amend_files:
        logging.info("Processing %s ...", amend_path.name)
        amendment_text = read_text_best_effort(amend_path)
        amendment_paras: list[Paragraph] = split_text_into_paragraphs(amendment_text)

        elementary_instructions = reviser.analyze_amendments(
            amendment_paragraphs=amendment_paras, template=template
        )

        # Write elementary instructions to a CSV with a stable schema
        elementary_path = (out_elementary_dir / amend_path.stem).with_suffix(".csv")
        # Normalize to records for pandas, ensuring consistent column order
        desired_columns = [
            "source_file",
            "amendment_start_line",
            "amendment_end_line",
            "target_section",
            "confidence_target",
            "change_explanation",
            "amendment_span_text",
        ]
        records = []
        for ins in elementary_instructions:
            try:
                item = ins.model_dump()
            except Exception:
                # Fallback if objects are not pydantic BaseModel-like
                item = {
                    "amendment_start_line": getattr(ins, "amendment_start_line", None),
                    "amendment_end_line": getattr(ins, "amendment_end_line", None),
                    "amendment_span_text": getattr(ins, "amendment_span_text", None),
                    "target_section": getattr(ins, "target_section", None),
                    "confidence_target": getattr(ins, "confidence_target", None),
                    "change_explanation": getattr(ins, "change_explanation", None),
                }
            item["source_file"] = amend_path.name
            records.append(item)

        if records:
            # Ensure columns order and include any extras at the end
            df = pd.DataFrame(records)
            cols = [c for c in desired_columns if c in df.columns] + [
                c for c in df.columns if c not in desired_columns
            ]
            df = df[cols]
        else:
            # Write headers only if no instructions
            df = pd.DataFrame(columns=desired_columns)
        df.to_csv(elementary_path, index=False)
        logging.info("wrote %s", elementary_path.relative_to(repo_root))


def run_step_2(
    repo_root: Path, model_name: str, template: ContractTypeTemplate, reviser: ContractReviser
) -> None:
    pass


def run_step_3(
    repo_root: Path, model_name: str, template: ContractTypeTemplate, reviser: ContractReviser
) -> None:
    pass


if __name__ == "__main__":
    main()
