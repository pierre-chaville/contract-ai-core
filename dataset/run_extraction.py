from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List
from datetime import datetime, timezone
from load_template import load_template
from contract_ai_core import (
    ContractTypeTemplate,
    DatapointExtractor,
    DatapointExtractorConfig,
    Paragraph,
    DocumentClassification,
    ClassifiedParagraph,
    split_text_into_paragraphs,
)


def build_classification_from_csv(csv_path: Path) -> DocumentClassification:
    cls_paragraphs: List[ClassifiedParagraph] = []
    clause_to_paragraphs: dict[str, List[int]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int((row.get("index") or "").strip())
            except Exception:
                continue
            text_i = (row.get("text") or "").strip()
            clause_key = (row.get("clause_key") or "").strip() or None
            conf_raw = (row.get("confidence") or "").strip()
            if conf_raw == "":
                confidence = None
            else:
                try:
                    confidence = float(conf_raw) / 100.0
                except Exception:
                    confidence = None
            cp = ClassifiedParagraph(
                paragraph=Paragraph(index=idx, text=text_i),
                clause_key=clause_key,
                confidence=confidence,
            )
            cls_paragraphs.append(cp)
            if clause_key:
                clause_to_paragraphs.setdefault(clause_key, []).append(idx)
    return DocumentClassification(paragraphs=cls_paragraphs, clause_to_paragraphs=clause_to_paragraphs or None)


def _write_tokens_usage(source_id: str, model: str, num_paragraphs: int, base_dir: Path) -> None:
    """Append a JSONL record for extraction calls (usage not available via LangChain here)."""
    try:
        out_dir = base_dir / "dataset" / "output" / "datapoints"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "tokens.jsonl"
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_id": source_id,
            "provider": "openai",
            "model": model,
            "num_paragraphs": num_paragraphs,
            "usage": {},
        }
        with out_path.open("a", encoding="utf-8") as f:
            import json as _json
            f.write(_json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract datapoints using gold classifications")
    parser.add_argument("--template", required=True, help="Template key (e.g., ISDA)")
    parser.add_argument("--model", required=True, help="Extractor model (e.g., gpt-4.1-mini)")
    args = parser.parse_args()

    template_key = args.template
    model_name = args.model

    repo_root = Path(__file__).resolve().parents[1]
    docs_dir = repo_root / "dataset" / "documents" / template_key
    gold_cls_dir = repo_root / "dataset" / "gold" / "clauses" / template_key
    output_dir = repo_root / "dataset" / "output" / "datapoints" / template_key / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config
    # temperature for gpt-5 is different from other models
    extractor_temperature = 1.0 if 'gpt-5' in model_name else 0.0    
    extractor = DatapointExtractor(
        DatapointExtractorConfig(
            provider="openai",
            model=model_name,
            temperature=extractor_temperature,
        )
    )

    template = ContractTypeTemplate.model_validate(load_template(template_key))

    md_files = sorted(docs_dir.glob("*.md"))
    if not md_files:
        print(f"No markdown files found in {docs_dir}")
        return

    # Map datapoint key -> title for CSV output
    key_to_title = {dp.key: dp.title for dp in template.datapoints}

    for doc_path in md_files:
        cls_path = gold_cls_dir / (doc_path.stem + ".csv")
        if not cls_path.exists():
            print(f"Skipping {doc_path.name}: missing classification {cls_path}")
            continue
        if os.path.exists(output_dir / (doc_path.stem + ".csv")):
            print(f"Skipping {doc_path.name} because it already exists")
            continue
        print(f"Extracting datapoints from {doc_path.name} using {cls_path.name} ...")

        text = doc_path.read_text(encoding="utf-8")
        _paragraphs: List[Paragraph] = split_text_into_paragraphs(text)
        classification = build_classification_from_csv(cls_path)
        extraction = extractor.extract(
            text=text,
            template=template,
            classified_clauses=classification,
        )

        # Log a usage record per document (token details not exposed via LangChain here)
        _write_tokens_usage(doc_path.name, model_name, len(_paragraphs), repo_root)

        out_path = output_dir / (doc_path.stem + ".csv")
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "title", "confidence", "value", "explanation"])
            for dp in extraction.datapoints:
                title = key_to_title.get(dp.key, "")
                confidence = dp.confidence
                confidence_percent = (
                    round(confidence * 100) if isinstance(confidence, (int, float)) else ""
                )
                writer.writerow([dp.key, title, confidence_percent, dp.value, getattr(dp, "explanation", "") or ""])
        print(f"  -> wrote {out_path}")


if __name__ == "__main__":
    main()
