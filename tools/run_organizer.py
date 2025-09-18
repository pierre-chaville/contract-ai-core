from __future__ import annotations

import argparse
import json
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
    args = parser.parse_args()

    model_name = args.model
    provider = args.provider

    repo_root = Path(__file__).resolve().parents[1]
    docs_dir = repo_root / "dataset" / "documents" / "organizer" / "files"
    pngs_dir = repo_root / "dataset" / "documents" / "organizer" / "pngs"
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

    # Determine which files to process: those without a per-document JSON already present
    to_process: list[tuple[str, list[Paragraph]]] = []
    images: list[bytes] = []
    for doc_path in txt_files:
        filename = doc_path.name
        out_json = output_dir / f"{doc_path.stem}.json"
        if not out_json.exists():
            text = doc_path.read_text(encoding="utf-8", errors="ignore")
            paragraphs = [Paragraph(text=p, index=i) for i, p in enumerate(text.split("\n"))]
            to_process.append((filename, paragraphs))
            # Look for a matching PNG preview: prefer <stem>.png, also try <name>.png
            png_candidates = [
                pngs_dir / f"{doc_path.stem}.png",
                pngs_dir / f"{doc_path.name}.png",
            ]
            png_bytes: bytes = b""
            for cand in png_candidates:
                if cand.exists():
                    try:
                        png_bytes = cand.read_bytes()
                        break
                    except Exception:
                        png_bytes = b""
                        break
            images.append(png_bytes)

    if not to_process:
        print("No pending files to process. All documents already have JSON outputs.")
    else:
        total = len(to_process)
        print("Processing", total, "files in batches of 10")
        for start in range(0, total, 10):
            end = min(start + 10, total)
            batch_docs = to_process[start:end]
            batch_imgs = images[start:end]
            print(f"  Batch {start + 1}-{end} of {total}")
            # Pass aligned images list; empty bytes mean no image for that document
            results = organizer.organize(batch_docs, images=batch_imgs)
            # Write one JSON per document in this batch
            for res in results:
                try:
                    data = res.model_dump()  # type: ignore[attr-defined]
                except Exception:
                    # Fallback manual dict if model_dump not available
                    def pack(field):
                        return {
                            "value": getattr(field, "value", None),
                            "confidence": getattr(field, "confidence", None),
                            "explanation": getattr(field, "explanation", None),
                        }

                    data = {
                        "filename": res.filename,
                        "contract_type": pack(res.contract_type),
                        "contract_type_version": pack(getattr(res, "contract_type_version", None)),
                        "contract_date": pack(res.contract_date),
                        "amendment_date": pack(res.amendment_date),
                        "amendment_number": pack(res.amendment_number),
                        "version_type": pack(res.version_type),
                        "status": pack(res.status),
                        "party_name_1": pack(getattr(res, "party_name_1", None)),
                        "party_role_1": pack(getattr(res, "party_role_1", None)),
                        "party_name_2": pack(getattr(res, "party_name_2", None)),
                        "party_role_2": pack(getattr(res, "party_role_2", None)),
                        "document_quality": pack(getattr(res, "document_quality", None)),
                    }
                out_json = output_dir / f"{Path(res.filename).stem}.json"
                with out_json.open("w", encoding="utf-8") as jf:
                    json.dump(data, jf, ensure_ascii=False, indent=2)
                print(f"    -> wrote {out_json}")


if __name__ == "__main__":
    main()
