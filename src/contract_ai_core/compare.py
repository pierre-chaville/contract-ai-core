from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field

from .schema import DocumentClassification, Paragraph
from .utilities import get_langchain_chat_model


@dataclass
class DocumentCompareConfig:
    provider: str = "openai"
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = 1600
    # Safety limits for context size
    max_pairs_for_llm: int = 50
    max_paragraph_chars: int = 1500
    # Cost for leaving a paragraph unmatched (insert/delete) in Hungarian matching, in [0,1]
    unmatched_cost: float = 0.6


class Severity(str):
    MATERIAL = "MATERIAL"
    IMPORTANT = "IMPORTANT"
    MINOR = "MINOR"
    NONE = "NONE"


class AlignmentResult(BaseModel):
    index_doc1: int | None = Field(
        description="Paragraph index in doc1 or null if inserted in doc2"
    )
    index_doc2: int | None = Field(
        description="Paragraph index in doc2 or null if deleted from doc1"
    )
    clause_key_doc1: str | None = Field(
        default=None, description="Clause key for doc1 paragraph if available"
    )
    text_doc1: str | None = Field(default=None, description="Paragraph text from doc1")
    text_doc2: str | None = Field(default=None, description="Paragraph text from doc2")
    diff: str | None = Field(
        default=None, description="Unified diff between aligned paragraphs (context-limited)"
    )
    text_markup: str | None = Field(
        default=None,
        description=(
            "Inline HTML markup of differences: insertions in red, deletions strikethrough red. "
            "Represents a consolidated view of changes from doc1 → doc2."
        ),
    )


class DiffAssessment(BaseModel):
    pair_index: int = Field(description="Index of the alignment pair being assessed")
    severity: str = Field(
        description=f"One of {Severity.MATERIAL}, {Severity.IMPORTANT}, {Severity.MINOR}, {Severity.NONE}"
    )
    rationale: str = Field(description="Short reasoning for the assigned severity")
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model confidence in [0,1] for the assigned severity",
    )


class CompareOutput(BaseModel):
    items: list[DiffAssessment]


class ComparisonItem(BaseModel):
    # AlignmentResult fields
    index_doc1: int | None
    index_doc2: int | None
    clause_key_doc1: str | None = None
    text_doc1: str | None = None
    text_doc2: str | None = None
    diff: str | None = None
    text_markup: str | None = None
    # Assessment fields
    severity: str | None = None
    rationale: str | None = None
    confidence: float | None = None


class CompareResult(BaseModel):
    items: list[ComparisonItem]


class DocumentCompare:
    """Compare two documents paragraph-by-paragraph and classify differences.

    Initialization:
    - paragraphs_doc1: list of Paragraph for the first document
    - paragraphs_doc2: list of Paragraph for the second document
    - classification_doc1: optional DocumentClassification for doc1 (used to surface clause key per paragraph)
    """

    def __init__(
        self,
        *,
        paragraphs_doc1: list[Paragraph],
        paragraphs_doc2: list[Paragraph],
        classification_doc1: DocumentClassification | None = None,
        config: DocumentCompareConfig | None = None,
    ) -> None:
        self.doc1: list[Paragraph] = paragraphs_doc1 or []
        self.doc2: list[Paragraph] = paragraphs_doc2 or []
        self.classification_doc1: DocumentClassification | None = classification_doc1
        self.config: DocumentCompareConfig = config or DocumentCompareConfig()

        self._llm = get_langchain_chat_model(
            provider=self.config.provider,
            model_name=self.config.model or "gpt-4.1-mini",
            temperature=float(self.config.temperature),
            max_tokens=self.config.max_tokens,
        )

        # Build quick lookup from doc1 paragraph index to clause_key
        self._index_to_clause_key: dict[int, str] = {}
        if self.classification_doc1:
            try:
                for cp in self.classification_doc1.paragraphs:
                    if cp.clause_key:
                        self._index_to_clause_key[cp.paragraph.index] = cp.clause_key
            except Exception:
                self._index_to_clause_key = {}

    def compare(self) -> CompareResult:
        # Step 1: align paragraphs (coarse diff)
        alignments = self._align_paragraphs(self.doc1, self.doc2)

        # Step 2: produce per-pair unified diffs
        enriched: list[AlignmentResult] = []
        for a in alignments:
            text1 = a.text_doc1 or ""
            text2 = a.text_doc2 or ""
            diff_txt = None
            markup_html = None
            if text1 or text2:
                diff_txt = self._unified_diff(text1, text2)
                markup_html = self._inline_markup(text1, text2)
            enriched.append(
                AlignmentResult(
                    index_doc1=a.index_doc1,
                    index_doc2=a.index_doc2,
                    clause_key_doc1=a.clause_key_doc1,
                    text_doc1=text1 if text1 else None,
                    text_doc2=text2 if text2 else None,
                    diff=diff_txt,
                    text_markup=markup_html,
                )
            )

        # Step 3: LLM assessment per pair (only for changed pairs)
        assessments_map = self._classify_differences(enriched)
        items: list[ComparisonItem] = []
        for i, a in enumerate(enriched):
            assess = assessments_map.get(i)
            items.append(
                ComparisonItem(
                    index_doc1=a.index_doc1,
                    index_doc2=a.index_doc2,
                    clause_key_doc1=a.clause_key_doc1,
                    text_doc1=a.text_doc1,
                    text_doc2=a.text_doc2,
                    diff=a.diff,
                    text_markup=a.text_markup,
                    severity=getattr(assess, "severity", None),
                    rationale=getattr(assess, "rationale", None),
                    confidence=getattr(assess, "confidence", None),
                )
            )
        return CompareResult(items=items)

    # ------------------------ Internals ------------------------

    def _align_paragraphs(
        self, doc1: list[Paragraph], doc2: list[Paragraph]
    ) -> list[AlignmentResult]:
        n1, n2 = len(doc1), len(doc2)
        if n1 == 0 and n2 == 0:
            return []
        # Try TF-IDF + Hungarian; fallback to difflib if unavailable
        try:
            import numpy as np  # type: ignore
            from scipy.optimize import linear_sum_assignment  # type: ignore
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
        except Exception:
            return self._align_paragraphs_fallback(doc1, doc2)

        texts1 = [p.text or "" for p in doc1]
        texts2 = [p.text or "" for p in doc2]
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            X = vec.fit_transform(texts1 + texts2)
        except Exception:
            return self._align_paragraphs_fallback(doc1, doc2)

        X1 = X[:n1]
        X2 = X[n1:]
        # Handle edge cases where one side is empty
        if n1 == 0:
            return [
                AlignmentResult(
                    index_doc1=None,
                    index_doc2=j,
                    clause_key_doc1=None,
                    text_doc1=None,
                    text_doc2=self._truncate(doc2[j].text),
                )
                for j in range(n2)
            ]
        if n2 == 0:
            return [
                AlignmentResult(
                    index_doc1=i,
                    index_doc2=None,
                    clause_key_doc1=self._index_to_clause_key.get(i),
                    text_doc1=self._truncate(doc1[i].text),
                    text_doc2=None,
                )
                for i in range(n1)
            ]

        sim = cosine_similarity(X1, X2)
        # Build square cost matrix with unmatched padding
        N = int(max(n1, n2))
        cost = np.full((N, N), float(self.config.unmatched_cost), dtype=float)
        cost[:n1, :n2] = 1.0 - sim
        if N > n1 and N > n2:
            cost[n1:, n2:] = 0.0

        row_ind, col_ind = linear_sum_assignment(cost)

        aligned: list[AlignmentResult] = []
        for r, c in zip(row_ind, col_ind, strict=False):
            if r < n1 and c < n2:
                ck = self._index_to_clause_key.get(r)
                aligned.append(
                    AlignmentResult(
                        index_doc1=r,
                        index_doc2=c,
                        clause_key_doc1=ck,
                        text_doc1=self._truncate(doc1[r].text),
                        text_doc2=self._truncate(doc2[c].text),
                    )
                )
            elif r < n1 and c >= n2:
                ck = self._index_to_clause_key.get(r)
                aligned.append(
                    AlignmentResult(
                        index_doc1=r,
                        index_doc2=None,
                        clause_key_doc1=ck,
                        text_doc1=self._truncate(doc1[r].text),
                        text_doc2=None,
                    )
                )
            elif r >= n1 and c < n2:
                aligned.append(
                    AlignmentResult(
                        index_doc1=None,
                        index_doc2=c,
                        clause_key_doc1=None,
                        text_doc1=None,
                        text_doc2=self._truncate(doc2[c].text),
                    )
                )
            else:
                # dummy-dummy — ignore
                pass

        aligned.sort(
            key=lambda a: (
                a.index_doc2 if a.index_doc2 is not None else 10**9,
                a.index_doc1 if a.index_doc1 is not None else 10**9,
            )
        )
        # for a in aligned:
        #     print("-" * 100)
        #     print(f"aligned: {a.index_doc1}, {a.index_doc2}, {a.text_doc1[:50]}, {a.text_doc2[:50]}")
        return aligned

    def _align_paragraphs_fallback(
        self, doc1: list[Paragraph], doc2: list[Paragraph]
    ) -> list[AlignmentResult]:
        seq1 = [p.text for p in doc1]
        seq2 = [p.text for p in doc2]
        sm = difflib.SequenceMatcher(a=seq1, b=seq2, autojunk=False)
        out: list[AlignmentResult] = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal" or tag == "replace":
                length = max(i2 - i1, j2 - j1)
                for k in range(length):
                    idx1 = i1 + k if (i1 + k) < i2 else None
                    idx2 = j1 + k if (j1 + k) < j2 else None
                    text1 = doc1[idx1].text if idx1 is not None else None
                    text2 = doc2[idx2].text if idx2 is not None else None
                    ck = self._index_to_clause_key.get(idx1) if idx1 is not None else None
                    out.append(
                        AlignmentResult(
                            index_doc1=idx1,
                            index_doc2=idx2,
                            clause_key_doc1=ck,
                            text_doc1=self._truncate(text1),
                            text_doc2=self._truncate(text2),
                        )
                    )
            elif tag == "delete":
                for idx in range(i1, i2):
                    text1 = doc1[idx].text
                    ck = self._index_to_clause_key.get(idx)
                    out.append(
                        AlignmentResult(
                            index_doc1=idx,
                            index_doc2=None,
                            clause_key_doc1=ck,
                            text_doc1=self._truncate(text1),
                            text_doc2=None,
                        )
                    )
            elif tag == "insert":
                for idx in range(j1, j2):
                    text2 = doc2[idx].text
                    out.append(
                        AlignmentResult(
                            index_doc1=None,
                            index_doc2=idx,
                            clause_key_doc1=None,
                            text_doc1=None,
                            text_doc2=self._truncate(text2),
                        )
                    )
        return out

    def _unified_diff(self, a: str, b: str) -> str:
        a_lines = (a or "").splitlines()
        b_lines = (b or "").splitlines()
        diff = difflib.unified_diff(a_lines, b_lines, fromfile="doc1", tofile="doc2", lineterm="")
        # Limit size to avoid token explosion
        lines: list[str] = []
        for i, line in enumerate(diff):
            if i > 400:
                lines.append("... (diff truncated) ...")
                break
            lines.append(line)
        return "\n".join(lines)

    def _inline_markup(self, a: str, b: str) -> str:
        """Produce inline HTML with insertions in red and deletions as red strikethrough.

        This uses a word-level diff via SequenceMatcher on tokenized text.
        """

        # Simple tokenization on whitespace to keep punctuation
        def tokenize(s: str) -> list[str]:
            # Split but keep separators to preserve spaces on join
            out: list[str] = []
            buf = ""
            for ch in s:
                if ch.isspace():
                    if buf:
                        out.append(buf)
                        buf = ""
                    out.append(ch)
                else:
                    buf += ch
            if buf:
                out.append(buf)
            return out

        a_tokens = tokenize(a or "")
        b_tokens = tokenize(b or "")
        sm = difflib.SequenceMatcher(a=a_tokens, b=b_tokens, autojunk=False)
        pieces: list[str] = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                pieces.append("".join(a_tokens[i1:i2]))
            elif tag == "delete":
                deleted = "".join(a_tokens[i1:i2])
                if deleted:
                    pieces.append(
                        f'<span style="color:#b71c1c;text-decoration:line-through;">{self._escape_html(deleted)}</span>'
                    )
            elif tag == "insert":
                inserted = "".join(b_tokens[j1:j2])
                if inserted:
                    pieces.append(
                        f'<span style="color:#b71c1c;">{self._escape_html(inserted)}</span>'
                    )
            elif tag == "replace":
                deleted = "".join(a_tokens[i1:i2])
                inserted = "".join(b_tokens[j1:j2])
                if deleted:
                    pieces.append(
                        f'<span style="color:#b71c1c;text-decoration:line-through;">{self._escape_html(deleted)}</span>'
                    )
                if inserted:
                    pieces.append(
                        f'<span style="color:#b71c1c;">{self._escape_html(inserted)}</span>'
                    )
        return "".join(pieces)

    def _escape_html(self, s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _truncate(self, text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        s = text
        limit = max(200, int(self.config.max_paragraph_chars))
        if len(s) > limit:
            return s[:limit] + " …"
        return s

    def _classify_differences(self, pairs: list[AlignmentResult]) -> dict[int, DiffAssessment]:
        # Consider only pairs with an actual diff
        def _has_change(p: AlignmentResult) -> bool:
            if not p.diff:
                return False
            # If unified diff contains any added/removed lines
            for ln in (p.diff or "").splitlines():
                if ln.startswith("+") or ln.startswith("-"):
                    # Ignore headers like +++/---
                    if ln.startswith("+++") or ln.startswith("---"):
                        continue
                    return True
            return False

        diff_items: list[tuple[int, AlignmentResult]] = [
            (i, p) for i, p in enumerate(pairs) if _has_change(p)
        ]
        if not diff_items:
            return {}

        # Limit how many pairs we send to the LLM
        max_pairs = max(1, int(self.config.max_pairs_for_llm))
        limited_items = diff_items[:max_pairs]

        # Prepare compact context for the LLM
        blocks: list[str] = []
        for orig_idx, p in limited_items:
            parts: list[str] = [f"PAIR {orig_idx}"]
            if p.clause_key_doc1:
                parts.append(f"CLAUSE: {p.clause_key_doc1}")
            parts.append(f"DOC1: {p.text_doc1 or ''}")
            parts.append(f"DOC2: {p.text_doc2 or ''}")
            # DIFF is guaranteed present by filter above
            parts.append("DIFF:\n" + (p.diff or ""))
            blocks.append("\n".join(parts))

        instruction = (
            "You are a senior legal analyst. Classify each paragraph pair difference by severity.\n"
            f"Severity levels: {Severity.MATERIAL}, {Severity.IMPORTANT}, {Severity.MINOR}, {Severity.NONE}.\n"
            "Guidance:\n"
            f"- {Severity.MATERIAL}: Changes that affect core obligations, rights, risk, pricing, termination, governing law, or liability caps.\n"
            f"- {Severity.IMPORTANT}: Non-core but significant terms (timelines, procedures, definitions) that may impact operations or risk.\n"
            f"- {Severity.MINOR}: Cosmetic edits, clarifications, formatting, typographical changes without substantive impact.\n"
            f"- {Severity.NONE}: No discernible difference.\n"
            "Return structured JSON with a list 'items', one per PAIR.\n"
            "For each item, include: pair_index, severity, rationale (<= 30 words), confidence in [0,1].\n"
        )

        prompt = instruction + "\n\n" + "\n\n".join(blocks)
        structured = self._llm.with_structured_output(
            CompareOutput, temperature=0.0, max_tokens=10000
        )  # type: ignore[arg-type]
        try:
            out: CompareOutput = structured.invoke(prompt)  # type: ignore[assignment]
            return {it.pair_index: it for it in out.items}
        except Exception:
            # Fallback: classify changed items as IMPORTANT with medium confidence
            import traceback

            print("Fallback classification", traceback.format_exc())
            items_map: dict[int, DiffAssessment] = {}
            for orig_idx, _ in limited_items:
                items_map[orig_idx] = DiffAssessment(
                    pair_index=orig_idx,
                    severity=Severity.IMPORTANT,
                    rationale="Fallback classification",
                    confidence=0.6,
                )
            return items_map
