from __future__ import annotations

import io
import json
import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from contract_ai_core.utilities import get_langchain_chat_model

try:  # Prefer LangChain's Pydantic v1 shim for compatibility
    from langchain_core.pydantic_v1 import BaseModel, Field  # type: ignore
except Exception:  # pragma: no cover
    from pydantic import BaseModel, Field  # type: ignore
from rapidfuzz import fuzz

DB_PATH = Path(__file__).resolve().parents[2] / "dataset" / "contracts.sqlite"
QUERIES_CSV = Path(__file__).resolve().parents[2] / "tools" / "legal_contract_queries.csv"
OUTPUT_QUERIES_DIR = Path(__file__).resolve().parents[2] / "dataset" / "output" / "queries"


@dataclass
class LLMDecision:
    is_db_query: bool
    search_strategy: str  # free-text explanation from the LLM
    sql: str | None
    graph_type: str | None  # one of {"bar", "line", "pie"} when graph requested
    explanation: str | None
    fuzzy_threshold: int | None = None
    render_types: Optional[List[str]] = None  # subset of {"graph", "table", "download"}
    select_fields: Optional[List[str]] = None  # preferred display fields/columns
    clauses_filter_expr: Optional[str] = None
    datapoints_filter_expr: Optional[str] = None
    fuzzy_fields_terms: Optional[Dict[str, List[str]]] = None


class PlanSchema(BaseModel):
    is_db_query: bool = Field(
        ..., description="Whether the question pertains to the CONTRACTS database"
    )
    search_strategy: str = Field(
        ..., description="Free-text explanation of the search approach chosen by the LLM"
    )
    sql: Optional[str] = Field(None, description="Safe SQL SELECT over CONTRACTS")
    render_types: Optional[List[Literal["graph", "table", "download"]]] = Field(
        None, description="Which renderings to show; multiple allowed"
    )
    select_fields: Optional[List[str]] = Field(
        None, description="Preferred list of fields/columns to display in the results"
    )
    graph_type: Optional[Literal["bar", "line", "pie"]] = Field(
        None,
        description="Chart type to use when render_types includes 'graph'",
    )
    explanation: Optional[str] = None
    fuzzy_threshold: Optional[int] = None
    clauses_filter_expr: Optional[str] = Field(
        None,
        description=(
            "Optional Python boolean expression evaluated per row when clause selection is required. "
            "Use variable list_clauses (list[str]) to reference row clause keys."
        ),
    )
    datapoints_filter_expr: Optional[str] = Field(
        None,
        description=(
            "Optional Python boolean expression evaluated per row when datapoint selection is required. "
            "Use variable list_datapoints (dict[str, Any]) to reference row datapoints."
        ),
    )
    fuzzy_fields_terms: Optional[Dict[str, List[str]]] = Field(
        None,
        description=(
            "Optional mapping of field name -> list of fuzzy terms to match after expressions. "
            "Example: {\"party_name_1\": [\"JP\"], \"full_text\": [\"events\", \"termination\"]}."
        ),
    )


SYSTEM_PRIMER = (
    "You are a helpful data assistant for a contracts database. "
    "The SQLite database has a single table named CONTRACTS. Column reference:\n"
    "| field | type | description | example |\n"
    "|------|------|-------------|---------|\n"
    "| contract_id | TEXT | Unique stable identifier | ISDA_000123 |\n"
    "| contract_number | TEXT | Business contract number | 2018-ISDA-45 |\n"
    "| contract_type | TEXT | Contract family key | ISDA |\n"
    "| contract_type_version | TEXT | Standard/version of the contract type | 2002 |\n"
    "| contract_date | DATE (YYYY-MM-DD) | Execution date of the contract | 2018-06-15 |\n"
    "| last_amendment_date | DATE (YYYY-MM-DD) | Date of most recent amendment | 2021-11-30 |\n"
    "| number_amendments | INTEGER | Count of amendments applied | 3 |\n"
    "| status | TEXT | Lifecycle status | Active |\n"
    "| party_name_1 | TEXT | Legal name of the first party | JPMorgan Chase Bank N.A. |\n"
    "| party_role_1 | TEXT | Role of the first party | Dealer |\n"
    "| party_name_2 | TEXT | Legal name of the second party | ACME Corp |\n"
    "| party_role_2 | TEXT | Role of the second party | Client |\n"
    "| department | TEXT | Internal department owning the contract | Treasury |\n"
    "| contract_owner | TEXT | Internal owner/responsible person | Jane Doe |\n"
    "| business_purpose | TEXT | Short description of business purpose | Hedging program |\n"
    "| full_text | TEXT | Full OCR/plaintext of the contract | ...long text... |\n"
    "| clauses_text | JSON | Map of clause_key -> clause text | {termination_event: '...'} |\n"
    "| list_clauses | JSON (array) | Clause keys present in the contract | [termination_event, setoff] |\n"
    "| datapoints | JSON (object) | Datapoint key -> value pairs | {governing_law: 'English'} |\n"
    "| guidelines | JSON | Review guidelines/flags or metadata | {risk: 'medium'} |\n"
    "The user will ask a question in English. Your job is to: "
    "2) Plan in three steps regardless of strategy: (a) SQL to filter by available scalar fields (do NOT rely on list_clauses or datapoints in SQL), (b) optional Python expressions to filter by list_clauses and datapoints, (c) optional fuzzy filtering on the resulting rows by specific fields. "
    "For full text searches in SQL, use LIKE with a :q parameter on full_text. "
    "Produce a safe SQL SELECT (avoid DDL/DML; no semicolons, no PRAGMAs). ALWAYS include contract_id in the SELECT so downstream steps can map results. Prefer adding a LIMIT 1000 unless the user explicitly asks for all rows. "
    'Provide optional "fuzzy_fields_terms" as a mapping of field -> list of fuzzy terms, and "fuzzy_threshold" (0-100, default 90). '
    "If the selection involves clauses or datapoints, include Python boolean expressions to post-filter rows: "
    "'clauses_filter_expr' uses variable list_clauses (list[str]) and should return True/False; "
    "'datapoints_filter_expr' uses variable list_datapoints (dict[str, Any]) and should return True/False. "
    "Examples: list_clauses and {'termination_event': 'Yes'}.get('key') style checks are valid. "
    "Set 'is_db_query' to true if the question pertains to querying the CONTRACTS table; otherwise false. "
    "Provide 'search_strategy' as a short free-text explanation of the overall approach (SQL filtering, expressions, fuzzy). "
    "Provide 'render_types' as an array of strings, each one of 'graph', 'table', or 'download'. Add 'graph' if the user asks for a chart, and 'table' if the user asks for a table. Use 'download' except if requested otherwise."
    "Return ONLY valid JSON with keys: is_db_query (bool), search_strategy (str), sql (str|null), render_types (array<string>|null), select_fields (array<string>|null), graph_type (str|null), explanation (str), fuzzy_fields_terms (object|null), fuzzy_threshold (int|nil), clauses_filter_expr (str|null), datapoints_filter_expr (str|null)."
)


def _load_isda_definitions() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load ISDA clauses and datapoints definitions from dataset/contract_types.

    Returns (clauses, datapoints), each a list of dicts with the requested fields.
    """
    base = Path(__file__).resolve().parents[2] / "dataset" / "contract_types"
    clauses_csv = base / "ISDA_clauses.csv"
    datapoints_csv = base / "ISDA_datapoints.csv"

    clauses: list[dict[str, Any]] = []
    datapoints: list[dict[str, Any]] = []

    try:
        if clauses_csv.exists():
            df_c = pd.read_csv(clauses_csv, encoding="utf-8").fillna("")
            df_c.columns = [str(c).strip().lstrip("\ufeff") for c in df_c.columns]
            for _, row in df_c.iterrows():
                key = row.get("key")
                title = row.get("title")
                desc = row.get("description")
                if key is None or title is None:
                    continue
                clauses.append(
                    {
                        "key": str(key),
                        "title": str(title),
                        "description": None if desc == "" else str(desc),
                    }
                )
    except Exception:
        clauses = []

    try:
        if datapoints_csv.exists():
            df_d = pd.read_csv(datapoints_csv, encoding="utf-8").fillna("")
            df_d.columns = [str(c).strip().lstrip("\ufeff") for c in df_d.columns]
            for _, row in df_d.iterrows():
                key = row.get("key")
                title = row.get("title")
                desc = row.get("description")
                dt = row.get("data_type")
                if key is None or title is None:
                    continue
                datapoints.append(
                    {
                        "key": str(key),
                        "title": str(title),
                        "description": None if desc == "" else str(desc),
                        "data_type": str(dt) if dt != "" else "",
                    }
                )
    except Exception:
        datapoints = []

    return clauses, datapoints


def _build_isda_reference_text() -> str:
    """Compose a compact ISDA reference section for the planning prompt."""
    clauses, datapoints = _load_isda_definitions()
    lines: list[str] = []
    if clauses:
        lines.append("ISDA CLAUSES (key | title | description):")
        for c in clauses:
            desc = c.get("description") or ""
            desc_short = desc.replace("\n", " ").strip()
            lines.append(f"- {c.get('key')} | {c.get('title')} | {desc_short}")
        lines.append("")
    if datapoints:
        lines.append("ISDA DATAPOINTS (key | title | description | data_type):")
        for d in datapoints:
            desc = d.get("description") or ""
            desc_short = desc.replace("\n", " ").strip()
            dtype = d.get("data_type") or ""
            lines.append(f"- {d.get('key')} | {d.get('title')} | {desc_short} | {dtype}")
        lines.append("")
    if not lines:
        return ""
    return "\n".join(lines)


def _build_contract_types_reference_text() -> str:
    """List available contract types with key, name, use_case, description.

    Prefer dataset/contract_types/lookup_values.csv (category == CONTRACT_TYPE). If unavailable,
    fall back to scanning individual JSON files.
    """
    base = Path(__file__).resolve().parents[2] / "dataset" / "contract_types"
    items: list[dict[str, Any]] = []

    # Preferred: lookup_values.csv
    try:
        lv_path = base / "lookup_values.csv"
        if lv_path.exists():
            df = pd.read_csv(lv_path, encoding="utf-8").fillna("")
            df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
            # Normalize and filter to CONTRACT_TYPE category
            df_cat = df[df["category"].astype(str).str.strip() == "CONTRACT_TYPE"].copy()
            for _, row in df_cat.iterrows():
                key = (row.get("key") or "").strip()
                label = (row.get("label") or "").strip()
                desc = row.get("description")
                meta = row.get("metadata")
                use_case_val = ""
                if meta not in (None, "", "nan", "NaN"):
                    try:
                        meta_obj = json.loads(meta) if not isinstance(meta, dict) else meta
                        if isinstance(meta_obj, dict):
                            use_case_val = str(meta_obj.get("use_case") or "").strip()
                    except Exception:
                        use_case_val = ""
                if key and label:
                    items.append(
                        {
                            "key": key,
                            "name": label,
                            "use_case": use_case_val,
                            "description": ("" if desc == "" else str(desc)).replace("\n", " ").strip(),
                        }
                    )
    except Exception:
        items = []

    # Fallback: scan JSON templates if no lookup records found
    if not items:
        try:
            for p in sorted(base.glob("*.json")):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(data, dict):
                    continue
                key = data.get("key") or p.stem
                name = data.get("name") or ""
                use_case = data.get("use_case") or ""
                desc = data.get("description") or ""
                items.append(
                    {
                        "key": str(key),
                        "name": str(name),
                        "use_case": str(use_case),
                        "description": str(desc).replace("\n", " ").strip(),
                    }
                )
        except Exception:
            items = []

    if not items:
        return ""
    # Stable sort by key for determinism
    items.sort(key=lambda x: x.get("key", ""))
    lines = ["CONTRACT TYPES (key | name | use_case | description):"]
    for it in items:
        lines.append(
            f"- {it.get('key')} | {it.get('name')} | {it.get('use_case')} | {it.get('description')}"
        )
    return "\n".join(lines)


def call_llm_plan(question: str) -> LLMDecision:
    model = get_langchain_chat_model(
        provider="openai", model_name="gpt-4.1", temperature=0.0, max_tokens=800
    )
    try:
        structured = model.with_structured_output(PlanSchema)  # type: ignore[attr-defined]
    except Exception:
        structured = None

    if structured is not None:
        primer = SYSTEM_PRIMER
        parts: list[str] = []
        ct_ref = _build_contract_types_reference_text()
        if ct_ref:
            parts.append(ct_ref)
        isda_ref = _build_isda_reference_text()
        if isda_ref:
            parts.append(isda_ref)
        ref_text = "\n\n".join(parts)
        print("ref_text", ref_text)
        if ref_text:
            primer = primer + "\n\n" + ref_text
        resp = structured.invoke(
            [
                {"role": "system", "content": primer},
                {"role": "user", "content": question},
            ]
        )  # type: ignore[attr-defined]
        plan: PlanSchema = resp  # type: ignore[assignment]
        print("plan", plan)
        return LLMDecision(
            is_db_query=bool(plan.is_db_query),
            search_strategy=str(plan.search_strategy),
            explanation=plan.explanation,
            sql=plan.sql,
            render_types=plan.render_types,
            graph_type=str(plan.graph_type) if plan.graph_type is not None else None,
            fuzzy_threshold=plan.fuzzy_threshold,
            select_fields=plan.select_fields,
            clauses_filter_expr=plan.clauses_filter_expr,
            datapoints_filter_expr=plan.datapoints_filter_expr,
            fuzzy_fields_terms=plan.fuzzy_fields_terms,
        )

    # Fallback to previous JSON parsing if structured output isn't available
    primer = SYSTEM_PRIMER
    parts: list[str] = []
    ct_ref = _build_contract_types_reference_text()
    if ct_ref:
        parts.append(ct_ref)
    isda_ref = _build_isda_reference_text()
    if isda_ref:
        parts.append(isda_ref)
    ref_text = "\n\n".join(parts)
    if ref_text:
        primer = primer + "\n\n" + ref_text
    print("primer", primer)
    messages = [
        {"role": "system", "content": primer},
        {"role": "user", "content": question},
    ]
    resp = model.invoke(messages)  # type: ignore[attr-defined]
    content = getattr(resp, "content", None) or getattr(resp, "text", None)
    if not isinstance(content, str):
        raise RuntimeError("LLM returned empty content")
    content_str = content.strip()
    if content_str.startswith("```"):
        content_str = content_str.strip("`")
    content_str = content_str.strip()
    try:
        obj = json.loads(content_str)
    except Exception:
        start = content_str.find("{")
        end = content_str.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(content_str[start : end + 1])
        else:
            raise

    return LLMDecision(
        is_db_query=bool(obj.get("is_db_query", True)),
        search_strategy=str(obj.get("search_strategy", "")),
        sql=(obj.get("sql") if obj.get("sql") is not None else None),
        graph_type=(str(obj.get("graph_type")) if obj.get("graph_type") is not None else None),
        explanation=obj.get("explanation"),
        fuzzy_threshold=(
            int(obj.get("fuzzy_threshold")) if obj.get("fuzzy_threshold") is not None else None
        ),
        render_types=(obj.get("render_types") if isinstance(obj.get("render_types"), list) else None),
        select_fields=(obj.get("select_fields") if isinstance(obj.get("select_fields"), list) else None),
        clauses_filter_expr=(obj.get("clauses_filter_expr") if isinstance(obj.get("clauses_filter_expr"), str) else None),
        datapoints_filter_expr=(obj.get("datapoints_filter_expr") if isinstance(obj.get("datapoints_filter_expr"), str) else None),
        fuzzy_fields_terms=(obj.get("fuzzy_fields_terms") if isinstance(obj.get("fuzzy_fields_terms"), dict) else None),
    )


def run_sql(sql: str, like_text: Optional[str] = None) -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    with sqlite3.connect(DB_PATH) as conn:
        # Optional parameter binding for LIKE patterns if the LLM leaves placeholders like :q
        params: Dict[str, Any] = {}
        if like_text is not None:
            params["q"] = f"%{like_text}%"
        return pd.read_sql_query(sql, conn, params=params)


def render_output(df: pd.DataFrame, output_type: str, outputs: Optional[List[str]] = None) -> None:
    if df.empty:
        st.info("No rows matched.")
        return

    # Determine which elements to show
    outputs_norm = [str(x).lower() for x in outputs] if outputs else None
    show_graph = outputs_norm is None or (
        (outputs_norm is not None)
        and ("graph" in outputs_norm or any(x in {"bar", "line", "pie"} for x in outputs_norm))
    )
    show_table = outputs_norm is None or ("table" in outputs_norm)
    show_download = outputs_norm is None or ("download" in outputs_norm)

    # Infer reasonable defaults for x, y, and an optional series (color) dimension
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    y_col = _infer_count_col(df) or (num_cols[0] if num_cols else None)

    # Choose x based on chart type with sensible preferences
    if output_type == "line":
        preferred_x = [
            "year",
            "contract_date",
            "contract_type",
            "status",
            "department",
            "party_name_1",
            "party_name_2",
        ]
    else:  # bar or other categorical-friendly charts
        preferred_x = [
            "year",
            "contract_date",
            "contract_type",
            "status",
            "department",
            "party_name_1",
            "party_name_2",
        ]
    x_col = _first_categorical_col(df, preferred=preferred_x)

    # Series (color) column: pick a different categorical with multiple levels
    series_preferences = ["contract_type", "status", "department", "party_name_1", "party_name_2"]
    series_col: Optional[str] = None
    for cand in series_preferences + [c for c in df.columns if c != x_col]:
        if cand == x_col or cand not in df.columns:
            continue
        try:
            if (
                pd.api.types.is_object_dtype(df[cand])
                or pd.api.types.is_string_dtype(df[cand])
                or cand in {"year", "contract_date"}
            ) and df[cand].nunique(dropna=True) > 1:
                series_col = cand
                break
        except Exception:
            continue

    if show_graph:
        if output_type == "bar" and y_col and x_col:
            fig = px.bar(df, x=x_col, y=y_col, color=series_col if series_col else None)
            if series_col:
                fig.update_layout(barmode="stack", showlegend=True, legend_title_text=series_col)
            st.plotly_chart(fig, use_container_width=True)
        elif output_type == "line" and y_col and x_col:
            fig = px.line(df, x=x_col, y=y_col, color=series_col if series_col else None)
            if series_col:
                fig.update_layout(showlegend=True, legend_title_text=series_col)
            st.plotly_chart(fig, use_container_width=True)
        elif output_type == "pie":
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if len(df.columns) >= 2 and num_cols:
                fig = px.pie(df, names=df.columns[0], values=num_cols[0])
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback: derive counts for a categorical column and plot by counts
                preferred_names = [
                    "status",
                    "contract_type",
                    "department",
                    "party_name_1",
                    "party_name_2",
                ]
                names_col = _first_categorical_col(df, preferred=preferred_names) or (
                    df.columns[0] if len(df.columns) > 0 else None
                )
                if names_col is not None:
                    counts = (
                        df[names_col]
                        .astype(str)
                        .value_counts(dropna=False)
                        .reset_index()
                    )
                    counts.columns = [names_col, "count"]
                    if not counts.empty:
                        fig = px.pie(counts, names=names_col, values="count")
                        st.plotly_chart(fig, use_container_width=True)
                    elif not show_table:
                        st.dataframe(df, use_container_width=True)
                elif not show_table:
                    st.dataframe(df, use_container_width=True)

    if show_table:
        st.dataframe(df, use_container_width=True)

    if show_download:
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download CSV", data=csv_buf.getvalue(), file_name="results.csv", mime="text/csv"
        )


def _compose_text_for_fuzzy(row: pd.Series) -> str:
    text_parts: List[str] = []
    if "full_text" in row and isinstance(row["full_text"], str):
        text_parts.append(row["full_text"])
    if "clauses" in row and row["clauses"] is not None:
        try:
            c = row["clauses"]
            if isinstance(c, str):
                c = json.loads(c)
            if isinstance(c, dict):
                text_parts.extend([str(v) for v in c.values() if v])
        except Exception:
            pass
    # Limit extremely long concatenations to keep scoring efficient
    return "\n".join(text_parts)[:200000]


STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "to",
    "for",
    "with",
    "in",
    "on",
    "by",
    "at",
    "as",
    "is",
    "are",
    "be",
    "been",
    "will",
    "shall",
    "from",
    "that",
    "this",
    "these",
    "those",
    "not",
    "no",
    "any",
    "all",
    "which",
}


def _normalize(s: str) -> str:
    return " ".join(
        "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in s).split()
    )


def _extract_terms_from_question(question: str, max_terms: int = 3) -> List[str]:
    q = question or ""
    # Prefer quoted phrases
    phrases: List[str] = []
    curr = []
    in_quote = False
    for ch in q:
        if ch in ('"', "'"):
            if in_quote and curr:
                phrases.append("".join(curr).strip())
                curr = []
            in_quote = not in_quote
            continue
        if in_quote:
            curr.append(ch)
    if phrases:
        return [p for p in phrases if p]
    # Fallback to keywords
    tokens = [t for t in _normalize(q).split() if len(t) >= 4 and t not in STOPWORDS]
    # Deduplicate preserving order
    seen = set()
    dedup: List[str] = []
    for t in tokens:
        if t not in seen:
            dedup.append(t)
            seen.add(t)
    return dedup[:max_terms] if dedup else ([_normalize(q)] if q.strip() else [])


def _fuzzy_score(text: str, terms: List[str]) -> int:
    if not text or not terms:
        return 0
    text_n = _normalize(text)
    # Score each term; require all terms to be present reasonably
    per_term_scores: List[int] = []
    for t in terms:
        t = _normalize(str(t))
        if not t:
            continue
        per_term_scores.append(
            max(
                fuzz.token_set_ratio(text_n, t),
                fuzz.partial_ratio(text_n, t),
            )
        )
    if not per_term_scores:
        return 0
    # Conservative: use the minimum across terms so all must match
    return min(per_term_scores)


def apply_fuzzy_filter(df: pd.DataFrame, terms: List[str], threshold: int) -> pd.DataFrame:
    if df.empty:
        return df
    # Ensure full_text present for scoring; if missing, fetch it
    need_full_text = "full_text" not in df.columns
    need_clauses = "clauses" not in df.columns
    if need_full_text or need_clauses:
        ids = (
            df["contract_id"].dropna().astype(str).unique().tolist()
            if "contract_id" in df.columns
            else []
        )
        if ids:
            placeholders = ",".join(["?"] * len(ids))
            with sqlite3.connect(DB_PATH) as conn:
                extra = pd.read_sql_query(
                    f"SELECT contract_id, full_text, clauses_text as clauses FROM CONTRACTS WHERE contract_id IN ({placeholders})",
                    conn,
                    params=ids,
                )
            df = df.merge(extra, on="contract_id", how="left")
    # Score and filter
    df = df.copy()
    df["fuzzy_text"] = df.apply(_compose_text_for_fuzzy, axis=1)
    df["fuzzy_score"] = df["fuzzy_text"].apply(lambda txt: _fuzzy_score(txt, terms))
    df = df[df["fuzzy_score"] >= threshold].sort_values(by="fuzzy_score", ascending=False)
    return df.drop(columns=["fuzzy_text"]) if "fuzzy_text" in df.columns else df


def _ensure_list_fields(df: pd.DataFrame, need_clauses: bool, need_datapoints: bool) -> pd.DataFrame:
    """Ensure dataframe has list_clauses and/or datapoints columns by fetching as needed.

    If missing and contract_id is available, fetch from DB and merge.
    """
    missing_clauses = need_clauses and ("list_clauses" not in df.columns)
    missing_datapoints = need_datapoints and ("datapoints" not in df.columns)
    if not (missing_clauses or missing_datapoints):
        return df
    if "contract_id" not in df.columns:
        st.warning(
            "Cannot apply clauses/datapoints filter because contract_id is not present in results."
        )
        return df
    ids = (
        df["contract_id"].dropna().astype(str).unique().tolist()
        if "contract_id" in df.columns
        else []
    )
    if not ids:
        return df
    placeholders = ",".join(["?"] * len(ids))
    select_cols = ["contract_id"]
    if missing_clauses:
        select_cols.append("list_clauses")
    if missing_datapoints:
        select_cols.append("datapoints")
    with sqlite3.connect(DB_PATH) as conn:
        extra = pd.read_sql_query(
            f"SELECT {', '.join(select_cols)} FROM CONTRACTS WHERE contract_id IN ({placeholders})",
            conn,
            params=ids,
        )
    return df.merge(extra, on="contract_id", how="left")


def _ensure_fields_for_fuzzy(df: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
    """Ensure dataframe has required fields for fuzzy matching by fetching as needed.

    If missing and contract_id is available, fetch from DB and merge.
    """
    missing = [f for f in fields if f not in df.columns]
    if not missing:
        return df
    if "contract_id" not in df.columns:
        st.warning("Cannot apply fuzzy field matching because contract_id is missing.")
        return df
    ids = df["contract_id"].dropna().astype(str).unique().tolist()
    if not ids:
        return df
    placeholders = ",".join(["?"] * len(ids))
    select_cols = ["contract_id"] + missing
    with sqlite3.connect(DB_PATH) as conn:
        extra = pd.read_sql_query(
            f"SELECT {', '.join(select_cols)} FROM CONTRACTS WHERE contract_id IN ({placeholders})",
            conn,
            params=ids,
        )
    return df.merge(extra, on="contract_id", how="left")


def apply_fuzzy_fields_filter(
    df: pd.DataFrame,
    fields_terms: Dict[str, List[str]],
    threshold: int,
) -> pd.DataFrame:
    """Apply fuzzy matching per specified fields with given terms.

    Requires each field's match score (min across its terms) to be >= threshold.
    Adds a 'fuzzy_score' column as the minimum score across all fields.
    """
    if df.empty or not fields_terms:
        return df
    needed_fields = sorted(set(fields_terms.keys()))
    df = _ensure_fields_for_fuzzy(df, needed_fields)

    def field_score(row: pd.Series, field: str, terms: List[str]) -> int:
        val = row.get(field)
        text = "" if val is None else str(val)
        return _fuzzy_score(text, terms)

    scores_per_field: Dict[str, List[int]] = {f: [] for f in needed_fields}
    combined: List[int] = []

    for _, row in df.iterrows():
        per_field_scores: List[int] = []
        for f in needed_fields:
            s = field_score(row, f, fields_terms.get(f, []))
            scores_per_field[f].append(s)
            per_field_scores.append(s)
        combined.append(min(per_field_scores) if per_field_scores else 0)

    df = df.copy()
    df["fuzzy_score"] = combined
    for f in needed_fields:
        df[f"fuzzy_score__{f}"] = scores_per_field[f]
    df = df[df["fuzzy_score"] >= threshold].sort_values(by="fuzzy_score", ascending=False)
    return df


def _safe_eval_expr(expr: str, list_clauses: Any, list_datapoints: Any) -> bool:
    """Evaluate a boolean expression with limited safe globals.

    Provides list_clauses (list) and list_datapoints (dict) plus a few safe builtins.
    Returns False on any exception.
    """
    try:
        safe_globals: Dict[str, Any] = {
            "__builtins__": {},
        }
        safe_locals: Dict[str, Any] = {
            "list_clauses": list_clauses,
            "list_datapoints": list_datapoints,
            # Selected safe helpers
            "len": len,
            "any": any,
            "all": all,
            "set": set,
            "sorted": sorted,
        }
        return bool(eval(expr, safe_globals, safe_locals))
    except Exception:
        return False


def apply_post_expressions(
    df: pd.DataFrame,
    clauses_expr: Optional[str],
    datapoints_expr: Optional[str],
) -> pd.DataFrame:
    """Apply optional Python expressions to filter rows using list_clauses/datapoints.

    - clauses_expr: expression using list_clauses (list[str])
    - datapoints_expr: expression using list_datapoints (dict[str, Any])
    """
    if df.empty:
        return df
    need_clauses = bool(clauses_expr)
    need_datapoints = bool(datapoints_expr)
    if not (need_clauses or need_datapoints):
        return df
    df = _ensure_list_fields(df, need_clauses, need_datapoints)

    def parse_json_val(val: Any, default: Any) -> Any:
        if val is None:
            return default
        if isinstance(val, (list, dict)):
            return val
        if isinstance(val, str):
            try:
                parsed = json.loads(val)
                if isinstance(default, list) and isinstance(parsed, list):
                    return parsed
                if isinstance(default, dict) and isinstance(parsed, dict):
                    return parsed
            except Exception:
                return default
        return default

    flags: list[bool] = []
    for _, row in df.iterrows():
        lc = parse_json_val(row.get("list_clauses"), [])
        dp = parse_json_val(row.get("datapoints"), {})
        ok = True
        if clauses_expr:
            ok = ok and _safe_eval_expr(clauses_expr, lc, dp)
        if datapoints_expr:
            ok = ok and _safe_eval_expr(datapoints_expr, lc, dp)
        flags.append(bool(ok))

    mask = pd.Series(flags, index=df.index)
    return df[mask]


def _infer_count_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["count", "counts", "cnt", "total", "n", "num", "value"]
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        col = lower_map.get(name)
        if col is not None and pd.api.types.is_numeric_dtype(df[col]):
            return col
    return None


def _first_categorical_col(df: pd.DataFrame, preferred: List[str]) -> Optional[str]:
    for p in preferred:
        if p in df.columns:
            return p
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            return c
    return None


def _build_results_context(df: pd.DataFrame) -> dict:
    context: Dict[str, Any] = {}
    count_col = _infer_count_col(df)

    # Row count reflects aggregation if present
    if count_col:
        try:
            context["row_count"] = int(
                pd.to_numeric(df[count_col], errors="coerce").fillna(0).sum()
            )
        except Exception:
            context["row_count"] = int(len(df))
    else:
        context["row_count"] = int(len(df))

    # Lightweight sample preview (non-aggregated columns preferred)
    preview_cols = [
        c
        for c in [
            "contract_id",
            "contract_type",
            "contract_type_version",
            "contract_date",
            "status",
            "party_name_1",
            "party_name_2",
            "fuzzy_score",
            count_col if count_col else None,
        ]
        if c in df.columns and c is not None
    ]
    if preview_cols:
        context["sample_rows"] = df[preview_cols].head(20).to_dict(orient="records")

    # Aggregates: contract_type
    if "contract_type" in df.columns:
        if count_col:
            # Use provided counts
            tmp = df[["contract_type", count_col]].copy()
            tmp[count_col] = pd.to_numeric(tmp[count_col], errors="coerce").fillna(0)
            grouped = (
                tmp.groupby("contract_type", dropna=False)[count_col]
                .sum()
                .sort_values(ascending=False)
            )
            context["contract_type_counts"] = {str(k): int(v) for k, v in grouped.head(20).items()}
        else:
            context["contract_type_counts"] = df["contract_type"].value_counts().head(20).to_dict()

    # Aggregates: year
    if "year" in df.columns and count_col:
        tmpy = df[["year", count_col]].copy()
        tmpy[count_col] = pd.to_numeric(tmpy[count_col], errors="coerce").fillna(0)
        grouped_y = tmpy.groupby("year", dropna=False)[count_col].sum().sort_values(ascending=False)
        context["year_counts"] = {str(k): int(v) for k, v in grouped_y.head(20).items()}
    elif "contract_date" in df.columns:
        years = df["contract_date"].dropna().astype(str).str.slice(0, 4)
        context["year_counts"] = years.value_counts().head(20).to_dict()

    if "fuzzy_score" in df.columns:
        context["fuzzy_score_stats"] = {
            "min": float(df["fuzzy_score"].min()),
            "max": float(df["fuzzy_score"].max()),
            "mean": float(df["fuzzy_score"].mean()),
        }
    return context


def explain_results_with_llm(
    question: str,
    decision: LLMDecision,
    sql: str,
    context: Dict[str, Any],
    *,
    provider: str = "openai",
    model_name: str = "gpt-4.1",
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> str:
    """Call the LLM to explain the results in plain language.

    Sends a compact context: query, strategy, fuzzy settings, SQL, row_count, aggregates, and up to 20 preview rows.
    """
    model = get_langchain_chat_model(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    system_msg = (
        "You are a helpful data analyst. Explain results clearly and concisely for a non-technical user. "
        "Summarize key findings, counts, trends by year or type if present, and call out caveats/limits. "
        "Do NOT paste the raw SQL. Also include a short section titled 'How results were selected' "
        "that explains, in plain language, the selection process based on the provided decision context "
        "(search strategy, any text LIKE usage, and fuzzy terms/threshold)."
    )

    # Build a compact decision context to guide the LLM's explanation of selection mechanics
    decision_context: Dict[str, Any] = {
        "search_strategy": str(decision.search_strategy),
        "render_types": decision.render_types,
        "select_fields": decision.select_fields,
        "graph_type": decision.graph_type,
        "fuzzy_fields_terms": decision.fuzzy_fields_terms,
        "fuzzy_threshold": decision.fuzzy_threshold,
        # Heuristic to hint at text LIKE usage without exposing SQL
        "used_text_like": (":q" in (sql or "")),
    }
    payload = {
        "question": question,
        "search_strategy": decision.search_strategy,
        "render_types": decision.render_types,
        "select_fields": decision.select_fields,
        "fuzzy_fields_terms": decision.fuzzy_fields_terms,
        "fuzzy_threshold": decision.fuzzy_threshold,
        "sql": sql,
        "results_context": context,
        "decision_context": decision_context,
    }
    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": (
                "Explain these query results and how they were selected:\n"
                + json.dumps(payload, ensure_ascii=False)
            ),
        },
    ]
    resp = model.invoke(messages)  # type: ignore[attr-defined]
    content = getattr(resp, "content", None) or getattr(resp, "text", None)
    return content if isinstance(content, str) else ""


def page() -> None:
    st.set_page_config(page_title="Database Query", page_icon="🗄️", layout="wide")
    st.title("Contracts Database Query")
    st.caption(
        "Ask questions in natural language. The assistant will plan and generate SQL over CONTRACTS."
    )

    if not DB_PATH.exists():
        st.error(
            f"Database not found at {DB_PATH} — run tools/run_create_db.py and tools/run_load_db.py first."
        )
        return

    with st.sidebar:
        st.subheader("Settings")
        provider = st.selectbox("Provider", ["openai", "azure", "anthropic"], index=0)
        model_name_default = (
            "gpt-4.1"
            if provider == "openai"
            else ("gpt-4o-mini" if provider == "azure" else "claude-3-5-sonnet-20240620")
        )
        model_name = st.text_input("Model name", value=model_name_default)

    # Load canned queries and provide navigation
    if "queries_list" not in st.session_state:
        try:
            if QUERIES_CSV.exists():
                dfq = pd.read_csv(QUERIES_CSV, encoding="utf-8").fillna("")
                # Try to detect a column named 'question' or first column
                if "question" in dfq.columns:
                    st.session_state["queries_list"] = dfq["question"].astype(str).tolist()
                else:
                    st.session_state["queries_list"] = dfq.iloc[:, 0].astype(str).tolist()
            else:
                st.session_state["queries_list"] = []
        except Exception:
            st.session_state["queries_list"] = []
    if "query_idx" not in st.session_state:
        st.session_state["query_idx"] = 0
    if "manual_mode" not in st.session_state:
        st.session_state["manual_mode"] = False

    # Navigation controls
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1, 6])
    with nav_col1:
        if st.button("Previous") and st.session_state["queries_list"]:
            st.session_state["manual_mode"] = False
            st.session_state["query_idx"] = max(0, st.session_state["query_idx"] - 1)
    with nav_col2:
        if st.button("Next") and st.session_state["queries_list"]:
            st.session_state["manual_mode"] = False
            st.session_state["query_idx"] = min(
                len(st.session_state["queries_list"]) - 1, st.session_state["query_idx"] + 1
            )
    with nav_col3:
        if st.button("New query"):
            st.session_state["manual_mode"] = True

    st.write("")
    # Determine initial text for the question input
    initial_q = ""
    if not st.session_state["manual_mode"] and st.session_state["queries_list"]:
        try:
            initial_q = st.session_state["queries_list"][st.session_state["query_idx"]]
        except Exception:
            initial_q = ""
    question = st.text_area(
        "Your question",
        value=initial_q,
        height=120,
        placeholder="e.g., Show top 10 counterparties by count for ISDA contracts signed after 2016",
    )

    # If a saved result exists for this question, show it directly
    if (question or "").strip():
        try:
            OUTPUT_QUERIES_DIR.mkdir(parents=True, exist_ok=True)
            qid_preview = hashlib.md5((question or "").strip().encode("utf-8")).hexdigest()
            saved_path = OUTPUT_QUERIES_DIR / f"{qid_preview}.json"
            if saved_path.exists():
                with saved_path.open("r", encoding="utf-8") as f:
                    saved_obj = json.load(f)
                saved_dec = saved_obj.get("decision", {})
                saved_rows = saved_obj.get("data", [])
                saved_expl = saved_obj.get("explanation", "")
                df_saved = pd.DataFrame(saved_rows)
                with st.expander("Saved decision", expanded=False):
                    st.markdown("**In DB scope**: " + str(saved_dec.get("is_db_query", "")))
                    st.markdown("**Search strategy**: " + str(saved_dec.get("search_strategy", "")))
                    st.markdown("**Explanation**: " + str(saved_obj.get("explanation") or saved_dec.get("explanation") or ""))
                    st.markdown("**SQL query**")
                    st.code(saved_dec.get("sql") or "")
                    rendering_list = saved_dec.get("render_types") or []
                    rendering_display = [
                        ("graph" if str(x).lower() in {"bar", "line", "pie"} else str(x))
                        for x in rendering_list
                    ]
                    st.markdown("**Rendering**: " + ", ".join(rendering_display))
                    st.markdown("**Graph type**: " + str(saved_dec.get("graph_type", "")))
                    sel_fields = saved_dec.get("select_fields") or []
                    st.markdown(
                        "**Display fields**: "
                        + (", ".join([str(x) for x in sel_fields]) if sel_fields else "")
                    )
                    if saved_dec.get("fuzzy_fields_terms") is not None:
                        st.markdown("**fuzzy_fields_terms**:")
                        try:
                            st.json(saved_dec.get("fuzzy_fields_terms"))
                        except Exception:
                            st.write(saved_dec.get("fuzzy_fields_terms"))
                    if saved_dec.get("clauses_filter_expr") or saved_dec.get("datapoints_filter_expr"):
                        st.markdown(
                            "**Clauses filter expression**: "
                            + str(saved_dec.get("clauses_filter_expr"))
                        )
                        st.markdown(
                            "**Datapoints filter expression**: "
                            + str(saved_dec.get("datapoints_filter_expr"))
                        )

                chart_type_saved = (saved_dec.get("graph_type") or "bar")
                render_output(df_saved, chart_type_saved, outputs=(saved_dec.get("render_types") or None))
                if saved_expl:
                    st.markdown("**Explanation**")
                    st.write(saved_expl)
        except Exception as e:
            st.warning("Failed to load saved results for this question.")
            st.exception(e)

    if st.button("Submit", type="primary") and question.strip():
        # First: ask LLM if question is in scope for the database
        with st.spinner("Checking question scope..."):
            try:
                # Lightweight scope check by leveraging the planner's boolean as primary signal
                scope_decision = call_llm_plan(question)
            except Exception as e:
                st.exception(e)
                return
        if not scope_decision.is_db_query:
            st.error("This question is out of scope for the contracts database.")
            with st.expander("Why not?"):
                st.write(scope_decision.explanation or scope_decision.search_strategy or "Not related to database content.")
            return

        with st.spinner("Planning with LLM..."):
            try:
                decision = call_llm_plan(question)
            except Exception as e:
                st.exception(e)
                return

        with st.expander("Decision (LLM plan)", expanded=False):
            st.markdown("**In DB scope**: " + str(decision.is_db_query))
            st.markdown("**Search strategy**: " + str(decision.search_strategy))
            st.markdown("**Explanation**: " + (decision.explanation or ""))
            st.markdown("**SQL query**")
            st.code(decision.sql)
            rendering_list = decision.render_types or []
            rendering_display = [
                ("graph" if str(x).lower() in {"bar", "line", "pie"} else str(x))
                for x in rendering_list
            ]
            st.markdown("**Rendering**: " + ", ".join(rendering_display))
            st.markdown("**Graph type**: " + str(decision.graph_type))
            st.markdown("**Display fields**: " + ", ".join(decision.select_fields or []) if decision.select_fields else "")
            # No global fuzzy terms; use fuzzy_fields_terms
            st.markdown("**fuzzy threshold**: " + str(decision.fuzzy_threshold))
            if decision.fuzzy_fields_terms is not None:
                st.markdown("**fuzzy_fields_terms**:")
                try:
                    st.json(decision.fuzzy_fields_terms)
                except Exception:
                    st.write(decision.fuzzy_fields_terms)
            if decision.clauses_filter_expr or decision.datapoints_filter_expr:
                st.markdown("**Clauses filter expression**: " + str(decision.clauses_filter_expr))
                st.markdown(
                    "**Datapoints filter expression**: " + str(decision.datapoints_filter_expr)
                )
        # Support LIKE with :q if present
        like_text: Optional[str] = None
        if decision.sql and ":q" in decision.sql:
            like_text = question

        if not decision.sql:
            st.error("The LLM did not produce a SQL query.")
            return

        with st.spinner("Running SQL..."):
            try:
                df = run_sql(decision.sql, like_text=like_text)
            except Exception as e:
                st.error("SQL execution failed.")
                st.code(decision.sql)
                st.exception(e)
                return

        # Optional Python expression post-filtering for clauses/datapoints
        if decision.clauses_filter_expr or decision.datapoints_filter_expr:
            with st.spinner("Applying clauses/datapoints filters..."):
                try:
                    df = apply_post_expressions(
                        df,
                        clauses_expr=decision.clauses_filter_expr,
                        datapoints_expr=decision.datapoints_filter_expr,
                    )
                except Exception as e:
                    st.warning("Post-expression filtering failed; showing current results.")
                    st.exception(e)

        # Optional fuzzy field-based filtering on current results
        if decision.fuzzy_fields_terms:
            threshold = decision.fuzzy_threshold or 75
            with st.spinner("Applying fuzzy field matching..."):
                try:
                    df = apply_fuzzy_fields_filter(df, decision.fuzzy_fields_terms, threshold)
                except Exception as e:
                    st.warning("Fuzzy field filtering failed; showing current results.")
                    st.exception(e)

        # Prepare display dataframe per select_fields (after all filtering)
        display_df = df
        if decision.select_fields:
            wanted = [c for c in decision.select_fields if c in display_df.columns]
            if wanted:
                display_df = display_df[wanted]
            else:
                st.warning("Requested select_fields not present in results; showing all columns.")

        chart_type = decision.graph_type or "bar"
        render_output(display_df, chart_type, outputs=decision.render_types)

        # LLM-based explanation of results
        with st.spinner("Explaining results..."):
            try:
                context = _build_results_context(df)
                explanation = explain_results_with_llm(
                    question=question,
                    decision=decision,
                    sql=decision.sql,
                    context=context,
                    provider=provider,
                    model_name=model_name,
                )
            except Exception as e:
                explanation = ""
                st.warning("Explanation generation failed.")
                st.exception(e)

        if explanation:
            st.markdown("**Explanation**")
            st.write(explanation)

        # Optional debug of the exact context sent to the explainer
        with st.expander("Debug: explanation context"):
            try:
                st.json(context)
            except Exception:
                st.write(context)

        # Persist query results to JSON in dataset/output/queries
        try:
            OUTPUT_QUERIES_DIR.mkdir(parents=True, exist_ok=True)
            qid = hashlib.md5((question or "").strip().encode("utf-8")).hexdigest()
            out_path = OUTPUT_QUERIES_DIR / f"{qid}.json"
            print(f"Saving query results to {out_path}")
            decision_dict = {
                "is_db_query": getattr(decision, "is_db_query", None),
                "search_strategy": decision.search_strategy,
                "render_types": decision.render_types,
                "select_fields": decision.select_fields,
                "graph_type": decision.graph_type,
                "fuzzy_fields_terms": decision.fuzzy_fields_terms,
                "fuzzy_threshold": decision.fuzzy_threshold,
                "sql": decision.sql,
            }
            data_rows: list[dict[str, Any]] = []
            try:
                data_rows = df.head(200).to_dict(orient="records")
            except Exception:
                data_rows = df.to_dict(orient="records")
            payload = {
                "question": question,
                "decision": decision_dict,
                "data": data_rows,
                "explanation": explanation,
            }
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            st.warning("Saving query results failed.")
            st.exception(e)


if __name__ == "__main__":  # For local single-file debugging
    page()
