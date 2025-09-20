from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from contract_ai_core import (
    ClassifiedParagraph,
    ClauseClassifier,
    ClauseClassifierConfig,
    ContractTypeTemplate,
    DocumentClassification,
    GuidelineReviewer,
    GuidelineReviewerConfig,
    Paragraph,
    text_to_paragraphs,
)
from utilities import load_template, write_tokens_usage


def build_classification_from_csv(csv_path: Path) -> DocumentClassification:
    cls_paragraphs: list[ClassifiedParagraph] = []
    clause_to_paragraphs: dict[str, list[int]] = {}
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
    return DocumentClassification(
        paragraphs=cls_paragraphs, clause_to_paragraphs=clause_to_paragraphs or None
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Review guidelines using gold classifications")
    parser.add_argument("--template", required=True, help="Template key (e.g., ISDA)")
    parser.add_argument("--model", required=True, help="Reviewer model (e.g., gpt-4.1-mini)")
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
    docs_dir = repo_root / "dataset" / "documents" / "contracts" / template_key
    cls_dir = repo_root / "dataset" / "output" / "clauses" / template_key / model_name
    cls_dir.mkdir(parents=True, exist_ok=True)
    output_dir = repo_root / "dataset" / "output" / "guidelines" / template_key / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config
    reviewer_temperature = 0.2
    reviewer = GuidelineReviewer(
        GuidelineReviewerConfig(
            provider=provider,
            model=model_name,
            temperature=reviewer_temperature,
            max_tokens=8000,
        )
    )

    template = ContractTypeTemplate.model_validate(load_template(template_key))

    txt_files = sorted(docs_dir.glob("*.txt"))
    if not txt_files:
        print(f"No text files found in {docs_dir}")
        return

    for doc_path in txt_files:
        cls_path = cls_dir / (doc_path.stem + ".csv")
        if os.path.exists(output_dir / (doc_path.stem + ".csv")):
            print(f"Skipping {doc_path.name} because it already exists")
            continue
        print(f"Reviewing guidelines for {doc_path.name} ...")

        text = doc_path.read_text(encoding="utf-8")
        _paragraphs: list[Paragraph] = text_to_paragraphs(text)
        if cls_path.exists():
            print(f"  -> using existing classification {cls_path.name}")
            classification = build_classification_from_csv(cls_path)
        else:
            print("  -> running classifier to generate classification CSV")
            cls_dir.mkdir(parents=True, exist_ok=True)
            classifier = ClauseClassifier(
                ClauseClassifierConfig(provider=provider, model=model_name)
            )
            classification, usage = classifier.classify_paragraphs_with_usage(
                _paragraphs, template, source_id=doc_path.name
            )
            write_tokens_usage(
                "clauses", doc_path.name, model_name, usage, len(_paragraphs), repo_root
            )
            with cls_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "clause_key", "confidence", "text"])
                for cp in classification.paragraphs:
                    conf_pct = round(cp.confidence * 100) if cp.confidence is not None else ""
                    writer.writerow(
                        [cp.paragraph.index, cp.clause_key or "", conf_pct, cp.paragraph.text]
                    )

        reviewed = reviewer.review(
            paragraphs=_paragraphs,
            template=template,
            classified_clauses=classification,
        )

        # Log a usage record per document (token details not exposed via LangChain here)
        write_tokens_usage(
            "guidelines", doc_path.name, model_name, None, len(_paragraphs), repo_root
        )

        out_path = output_dir / (doc_path.stem + ".csv")
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "key",
                    "fallback_from_key",
                    "priority",
                    "guideline",
                    "guideline_matched",
                    "confidence",
                    "evidence",
                    "explanation",
                ]
            )
            # Use template order for reproducibility
            tmpl_by_key = {g.key: g for g in template.guidelines}
            for r in reviewed:
                g = tmpl_by_key.get(r.key)
                guideline_text = g.guideline if g else ""
                priority = g.priority if g else ""
                conf_pct = (
                    round(r.confidence * 100) if isinstance(r.confidence, (int, float)) else ""
                )
                evidence = (
                    r.evidence_paragraph_indices
                    if getattr(r, "evidence_paragraph_indices", None)
                    else []
                )
                writer.writerow(
                    [
                        r.key,
                        g.fallback_from_key or "",
                        priority,
                        guideline_text,
                        r.guideline_matched,
                        conf_pct,
                        evidence,
                        r.explanation or "",
                    ]
                )
        print(f"  -> wrote {out_path}")


if __name__ == "__main__":
    main()
