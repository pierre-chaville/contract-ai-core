from __future__ import annotations

"""CLI: Run scope filtering for contracts based on gold metadata.

For each contract file under dataset/documents/contracts/<TYPE>/*.md:
- Read gold organizer metadata dataset/gold/organizer/<filename>.json to get contract_type
- If dataset/contract_types/<contract_type>.json exists, load the template
- Use DocumentFilter to locate spans for each template.filtering_scope
- Write results to dataset/output/filter/<filename>.json

Usage:
  python tools/run_filter.py --model gpt-4.1-mini [--provider openai]
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from contract_ai_core import ContractTypeTemplate, DocumentFilter, DocumentFilterConfig
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


def find_gold_contract_type(repo_root: Path, stem: str) -> tuple[Optional[str], Optional[str]]:
    gold_path = repo_root / "dataset" / "gold" / "organizer" / f"{stem}.json"
    if not gold_path.exists():
        return None, None
    try:
        data = json.loads(gold_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    ct = (data.get("contract_type") or {}).get("value")
    version_type = (data.get("version_type") or {}).get("value")
    return (ct, version_type) if isinstance(ct, str) and ct.strip() else (None, version_type)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Run scope filtering for contracts")
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "azure", "anthropic"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name for DocumentFilter (e.g., gpt-4.1-mini)",
    )
    args = parser.parse_args()

    provider = args.provider
    model_name = args.model

    repo_root = Path(__file__).resolve().parents[1]

    contracts_root = repo_root / "dataset" / "documents" / "organizer" / "files"

    # Iterate over all known contract files (*.md) grouped by type folder
    if not contracts_root.exists():
        logging.warning("Contracts directory not found: %s", contracts_root)
        return

    # Initialize filter
    doc_filter = DocumentFilter(DocumentFilterConfig(provider=provider, model=model_name))

    for doc_path in sorted(contracts_root.glob("*.txt")):
        stem = doc_path.stem
        # Determine contract type from gold metadata (by stem)
        contract_type, version_type = find_gold_contract_type(repo_root, stem)
        if not contract_type:
            logging.info("Skipping %s: missing gold contract_type", doc_path.name)
            continue
        if version_type == "AMENDMENT":
            logging.info("Skipping %s: version_type is AMENDMENT", doc_path.name)
            continue

        # Load template if present under dataset/contract_types
        try:
            template_dict = load_template(contract_type)
        except Exception:
            logging.info(
                "Skipping %s: template not found for contract_type=%s", doc_path.name, contract_type
            )
            continue
        try:
            template = ContractTypeTemplate.model_validate(template_dict)
        except Exception as exc:
            logging.warning("Invalid template for %s: %r", contract_type, exc)
            continue
        if not template.filtering_scopes:
            logging.info("No filtering_scopes for %s; skipping %s", contract_type, doc_path.name)
            continue

        out_dir = repo_root / "dataset" / "output" / "filter" / contract_type / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}.json"
        if out_path.exists():
            logging.info("Skipping %s (already exists)", out_path.name)
            continue

        logging.info("Filtering %s (type=%s) ...", doc_path.name, contract_type)
        text = read_text_best_effort(doc_path)

        name_to_span = doc_filter.locate_template_scopes(document_text=text, template=template)

        # Serialize to a stable JSON structure
        result = {
            "filename": doc_path.name,
            "contract_type": contract_type,
            "scopes": [
                {
                    "name": name,
                    "start_line": span[0],
                    "end_line": span[1],
                    "confidence": span[2],
                    "explanation": span[3],
                }
                for name, span in name_to_span.items()
            ],
        }

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logging.info("wrote %s", out_path.relative_to(repo_root))


if __name__ == "__main__":
    main()
