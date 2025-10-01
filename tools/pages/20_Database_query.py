from __future__ import annotations

import io
import json
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


@dataclass
class LLMDecision:
    is_db_query: bool
    search_strategy: str  # one of {"sql_only", "text_fulltext", "fuzzy", "hybrid"}
    sql: str | None
    graph_type: str | None  # one of {"bar", "line", "pie"} when graph requested
    explanation: str | None
    fuzzy_terms: List[str] | None = None
    fuzzy_threshold: int | None = None
    outputs: List[str] | None = None  # subset of {"graph", "table", "download"}


class PlanSchema(BaseModel):
    is_db_query: bool = Field(..., description="Whether the question pertains to the database")
    search_strategy: Literal["sql_only", "text_fulltext", "fuzzy", "hybrid"]
    sql: Optional[str] = Field(None, description="Safe SQL SELECT over CONTRACTS")
    outputs: Optional[List[Literal["graph", "table", "download"]]] = Field(
        None, description="Which renderings to show; multiple allowed"
    )
    graph_type: Optional[Literal["bar", "line", "pie"]] = Field(
        None,
        description="Chart type to use when outputs includes 'graph'",
    )
    explanation: Optional[str] = None
    fuzzy_terms: Optional[List[str]] = None
    fuzzy_threshold: Optional[int] = None


SYSTEM_PRIMER = (
    "You are a helpful data assistant for a contracts database. "
    "The SQLite database has a single table named CONTRACTS with columns: "
    "contract_id, contract_number, contract_type, contract_type_version, contract_date, last_amendment_date, "
    "number_amendments, status, party_name_1, party_role_1, party_name_2, party_role_2, department, contract_owner, "
    "business_purpose, full_text, clauses_text (JSON), list_clauses (JSON array), datapoints (JSON), guidelines (JSON). "
    "The user will ask a question in English. Your job is to: "
    "1) decide if the request is about this database; "
    "2) choose a search strategy (sql_only, text_fulltext, fuzzy, hybrid). full_text searches must use SQL LIKE on full_text with a :q parameter; "
    "3) produce a safe SQL SELECT query over CONTRACTS (avoid DDL/DML; no semicolons, no PRAGMAs); "
    "4) choose visualization based on the user's request: set 'outputs' to an array subset of ['graph','table','download'] indicating which renderings to show (multiple allowed); if 'graph' is included, set 'graph_type' to one of ['bar','line','pie']. "
    "5) provide a detailed explanation of your reasoning. "
    "Fuzzy search guidance: when strategy is fuzzy or hybrid, FIRST generate an SQL that NARROWS candidates using structured filters from the question (e.g., contract_type, contract_date range/year, status). "
    "ALWAYS include contract_id in the SELECT so downstream fuzzy matching can map results. "
    "Prefer adding a LIMIT 1000 unless the user explicitly asks for all rows. "
    "Treat list_clauses as a JSON array of clause titles. Use this field to generate a clauses column for fuzzy matching."
    "Do not use JSON extraction functions; treat clauses_text/datapoints/guidelines as opaque JSON and let the fuzzy step match their text. "
    'Also provide optional "fuzzy_terms" (array of 1-3 short phrases) and "fuzzy_threshold" (int 0-100, default 90). '
    "Return ONLY valid JSON with keys: is_db_query (bool), search_strategy (str), sql (str|null), outputs (array<string>|null), graph_type (str|null), explanation (str), fuzzy_terms (array<string>|null), fuzzy_threshold (int|nil)."
)


def call_llm_plan(question: str) -> LLMDecision:
    model = get_langchain_chat_model(
        provider="openai", model_name="gpt-4.1", temperature=0.0, max_tokens=800
    )
    try:
        structured = model.with_structured_output(PlanSchema)  # type: ignore[attr-defined]
    except Exception:
        structured = None

    if structured is not None:
        resp = structured.invoke(
            [
                {"role": "system", "content": SYSTEM_PRIMER},
                {"role": "user", "content": question},
            ]
        )  # type: ignore[attr-defined]
        plan: PlanSchema = resp  # type: ignore[assignment]
        return LLMDecision(
            is_db_query=bool(plan.is_db_query),
            search_strategy=str(plan.search_strategy),
            sql=plan.sql,
            graph_type=str(plan.graph_type) if plan.graph_type is not None else None,
            explanation=plan.explanation,
            fuzzy_terms=plan.fuzzy_terms,
            fuzzy_threshold=plan.fuzzy_threshold,
            outputs=plan.outputs,
        )

    # Fallback to previous JSON parsing if structured output isn't available
    messages = [
        {"role": "system", "content": SYSTEM_PRIMER},
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
        search_strategy=str(obj.get("search_strategy", "sql_only")),
        sql=(obj.get("sql") if obj.get("sql") is not None else None),
        graph_type=(str(obj.get("graph_type")) if obj.get("graph_type") is not None else None),
        explanation=obj.get("explanation"),
        fuzzy_terms=(obj.get("fuzzy_terms") if isinstance(obj.get("fuzzy_terms"), list) else None),
        fuzzy_threshold=(
            int(obj.get("fuzzy_threshold")) if obj.get("fuzzy_threshold") is not None else None
        ),
        outputs=(obj.get("outputs") if isinstance(obj.get("outputs"), list) else None),
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
    show_graph = outputs_norm is None or ("graph" in outputs_norm)
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
        "Do NOT restate the raw SQL; focus on what the results mean."
    )
    payload = {
        "question": question,
        "search_strategy": decision.search_strategy,
        "fuzzy_terms": decision.fuzzy_terms,
        "fuzzy_threshold": decision.fuzzy_threshold,
        "sql": sql,
        "results_context": context,
    }
    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": ("Explain these query results:\n" + json.dumps(payload, ensure_ascii=False)),
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

    st.write("")
    question = st.text_area(
        "Your question",
        height=120,
        placeholder="e.g., Show top 10 counterparties by count for ISDA contracts signed after 2016",
    )

    if st.button("Submit", type="primary") and question.strip():
        with st.spinner("Planning with LLM..."):
            try:
                decision = call_llm_plan(question)
            except Exception as e:
                st.exception(e)
                return

        if not decision.is_db_query:
            st.warning("This question is out of scope for the contracts database.")
            return

        with st.expander("Decision (LLM plan)", expanded=False):
            st.markdown("**is_db_query**: " + str(decision.is_db_query))
            st.markdown("**search_strategy**: " + str(decision.search_strategy))
            st.markdown("**SQL**")
            st.code(decision.sql)
            st.markdown("**graph_type**: " + str(decision.graph_type))
            st.markdown("**outputs**: " + str(decision.outputs))
            st.markdown("**explanation**: " + (decision.explanation or ""))
            st.markdown("**fuzzy_terms**: " + str(decision.fuzzy_terms))
            st.markdown("**fuzzy_threshold**: " + str(decision.fuzzy_threshold))
        # If the plan requests full text search, hint the LLM expects LIKE with :q
        like_text: Optional[str] = None
        if (
            decision.search_strategy in {"text_fulltext", "hybrid"}
            and decision.sql
            and ":q" in decision.sql
        ):
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

        # Optional fuzzy post-filtering
        if decision.search_strategy in {"fuzzy", "hybrid"}:
            terms = decision.fuzzy_terms or _extract_terms_from_question(question)
            threshold = decision.fuzzy_threshold or 75
            with st.spinner("Applying fuzzy matching..."):
                try:
                    df = apply_fuzzy_filter(df, terms=terms, threshold=threshold)
                except Exception as e:
                    st.warning("Fuzzy filtering failed; showing raw SQL results.")
                    st.exception(e)

        chart_type = decision.graph_type or "bar"
        render_output(df, chart_type, outputs=decision.outputs)

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


if __name__ == "__main__":  # For local single-file debugging
    page()
