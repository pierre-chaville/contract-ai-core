from __future__ import annotations

import argparse
import csv
from pathlib import Path

from contract_ai_core import (
    ContractOrganizer,
    ContractOrganizerConfig,
    Paragraph,
)
from utilities import load_lookup_values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize and extract contract metadata for documents"
    )
    parser.add_argument("--model", required=True, help="Model to use (e.g., gpt-4.1-mini)")
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "azure", "anthropic"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--source_dir",
        default="test",
        help="Optional override for documents directory (defaults to dataset/documents/organizer)",
    )
    args = parser.parse_args()

    model_name = args.model
    provider = args.provider

    repo_root = Path(__file__).resolve().parents[1]
    docs_dir = repo_root / "dataset" / "documents" / "organizer" / args.source_dir
    output_dir = repo_root / "dataset" / "output" / "organizer" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(docs_dir.glob("*.txt"))
    if not txt_files:
        print(f"No text files found in {docs_dir}")
        return

    # temperature for gpt-5 is different from other models
    temperature = 1.0 if "gpt-5" in model_name else 0.2

    organizer = ContractOrganizer(
        ContractOrganizerConfig(
            provider=provider,
            model=model_name,
            temperature=temperature,
            max_tokens=8000,
            lookup_contract_types=load_lookup_values("CONTRACT_TYPE"),
            lookup_version_types=load_lookup_values("VERSION_TYPE"),
            lookup_statuses=load_lookup_values("STATUS"),
        )
    )

    # Read existing results if present
    out_path = output_dir / "results.csv"
    existing_rows: dict[str, dict[str, str]] = {}
    header = [
        "filename",
        "contract_type",
        "contract_type_confidence",
        "contract_type_explanation",
        "contract_date",
        "contract_date_confidence",
        "contract_date_explanation",
        "amendment_date",
        "amendment_date_confidence",
        "amendment_date_explanation",
        "amendment_number",
        "amendment_number_confidence",
        "amendment_number_explanation",
        "version_type",
        "version_type_confidence",
        "version_type_explanation",
        "status",
        "status_confidence",
        "status_explanation",
    ]

    if out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fn = (row.get("filename") or "").strip()
                    if not fn:
                        continue
                    existing_rows[fn] = {k: (row.get(k) or "") for k in header}
        except Exception:
            existing_rows = {}

    # Determine which files to process (missing or empty contract_type)
    to_process: list[tuple[str, list[Paragraph]]] = []
    for doc_path in txt_files:
        filename = doc_path.name
        prev = existing_rows.get(filename)
        if prev is None or (prev.get("contract_type", "").strip() == ""):
            text = doc_path.read_text(encoding="utf-8", errors="ignore")
            paragraphs = [Paragraph(text=p, index=i) for i, p in enumerate(text.split("\n"))]
            to_process.append((filename, paragraphs))

    if not to_process:
        print("No pending files to process. Keeping existing results.")
    else:
        print("Processing", len(to_process), "files")
        results = organizer.organize(to_process)

        def pct(x: float | None) -> str:
            try:
                return str(int(round(float(x) * 100))) if x is not None else ""
            except Exception:
                return ""

        # Merge new results into existing rows
        for res in results:
            existing_rows[res.filename] = {
                "filename": res.filename,
                "contract_type": res.contract_type.value or "",
                "contract_type_confidence": pct(res.contract_type.confidence),
                "contract_type_explanation": res.contract_type.explanation or "",
                "contract_date": res.contract_date.value or "",
                "contract_date_confidence": pct(res.contract_date.confidence),
                "contract_date_explanation": res.contract_date.explanation or "",
                "amendment_date": res.amendment_date.value or "",
                "amendment_date_confidence": pct(res.amendment_date.confidence),
                "amendment_date_explanation": res.amendment_date.explanation or "",
                "amendment_number": res.amendment_number.value or "",
                "amendment_number_confidence": pct(res.amendment_number.confidence),
                "amendment_number_explanation": res.amendment_number.explanation or "",
                "version_type": res.version_type.value or "",
                "version_type_confidence": pct(res.version_type.confidence),
                "version_type_explanation": res.version_type.explanation or "",
                "status": res.status.value or "",
                "status_confidence": pct(res.status.confidence),
                "status_explanation": res.status.explanation or "",
            }

    # Write back the full CSV
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for fn in sorted(existing_rows.keys()):
            writer.writerow(existing_rows[fn])
    print(f"  -> wrote {out_path}")


if __name__ == "__main__":
    main()
