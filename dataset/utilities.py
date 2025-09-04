import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def load_template(template_key: str) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    folder_path = repo_root / "dataset" / "contract_types"
    json_path = folder_path / f"{template_key}.json"
    csv_path_clauses = folder_path / f"{template_key}_clauses.csv"
    csv_path_enums = folder_path / f"{template_key}_enums.csv"
    csv_path_datapoints = folder_path / f"{template_key}_datapoints.csv"
    csv_path_guidelines = folder_path / f"{template_key}_guidelines.csv"

    model = json.loads(open(json_path, encoding="utf-8").read())
    model["clauses"] = []
    model["enums"] = []
    model["datapoints"] = []
    model["guidelines"] = []

    # Read CSV with tolerant decoding
    df = pd.read_csv(csv_path_clauses, encoding="utf-8")

    # Normalize column names (strip whitespace and BOM artefacts)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    for _, row in df.iterrows():
        clause_key = row.get("key")
        clause_title = row.get("title")
        clause_description = row.get("description")

        if clause_key is None or clause_title is None:
            continue

        model["clauses"].append(
            {
                "key": str(clause_key),
                "title": str(clause_title),
                "description": None if pd.isna(clause_description) else str(clause_description),
            }
        )

    # Read CSV with tolerant decoding
    df_enums = pd.read_csv(csv_path_enums, encoding="utf-8")

    # Normalize column names (strip whitespace and BOM artefacts)
    df_enums.columns = [str(c).strip().lstrip("\ufeff") for c in df_enums.columns]

    enums = {}
    for _, row in df_enums.iterrows():
        enum_key = row.get("key")
        enum_title = row.get("title")
        if enum_key is None:
            continue

        if enum_key not in enums:
            enums[enum_key] = {
                "key": enum_key,
                "title": enum_title,
                "options": [],
            }

        enums[enum_key]["options"].append(
            {
                "code": row.get("code"),
                "description": row.get("description"),
            }
        )

    for enum_key, enum in enums.items():
        model["enums"].append(
            {"key": str(enum_key), "title": str(enum["title"]), "options": enum["options"]}
        )

    # Read CSV with tolerant decoding
    df = pd.read_csv(csv_path_datapoints, encoding="utf-8")

    # Normalize column names (strip whitespace and BOM artefacts)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    for _, row in df.iterrows():
        datapoint_key = row.get("key")
        datapoint_title = row.get("title")
        datapoint_description = row.get("description")
        datapoint_data_type = row.get("data_type")
        datapoint_enum_key = row.get("enum_key")
        datapoint_enum_multi_select = row.get("enum_multi_select")
        datapoint_scope = row.get("scope")
        datapoint_clause_keys = row.get("clause_keys")

        if datapoint_key is None or datapoint_title is None:
            continue

        model["datapoints"].append(
            {
                "key": str(datapoint_key),
                "title": str(datapoint_title),
                "description": (
                    None if pd.isna(datapoint_description) else str(datapoint_description)
                ),
                "data_type": str(datapoint_data_type),
                "enum_key": str(datapoint_enum_key),
                "enum_multi_select": bool(datapoint_enum_multi_select),
                "scope": str(datapoint_scope),
                "clause_keys": (
                    None
                    if pd.isna(datapoint_clause_keys)
                    else str(datapoint_clause_keys).split(",")
                ),
            }
        )

    # Read CSV with tolerant decoding
    df_guidelines = pd.read_csv(csv_path_guidelines, encoding="utf-8")

    # Normalize column names (strip whitespace and BOM artefacts)
    df_guidelines.columns = [str(c).strip().lstrip("\ufeff") for c in df_guidelines.columns]

    for idx, row in df_guidelines.iterrows():
        raw_id = row.get("id")
        fallback_from_key = row.get("fallback_from_key")
        guideline_text = row.get("guideline")
        raw_priority = row.get("priority")
        raw_scope = row.get("scope")
        raw_clause_keys = row.get("clause_keys")

        # Skip if no primary guideline text
        if guideline_text is None or (
            isinstance(guideline_text, float) and pd.isna(guideline_text)
        ):
            continue

        # Generate a stable key if missing/blank id
        if (
            raw_id is None
            or (isinstance(raw_id, float) and pd.isna(raw_id))
            or str(raw_id).strip() == ""
        ):
            guideline_key = f"guideline_{idx + 1}"
        else:
            guideline_key = str(raw_id)

        if (
            fallback_from_key is None
            or (isinstance(fallback_from_key, float) and pd.isna(fallback_from_key))
            or str(fallback_from_key).strip() == ""
            or str(fallback_from_key).strip().lower() == "nan"
        ):
            fallback_from_key = None

        # Defaults for priority and scope if missing
        priority_value = (
            "medium"
            if raw_priority is None
            or (isinstance(raw_priority, float) and pd.isna(raw_priority))
            or str(raw_priority).strip() == ""
            else str(raw_priority)
        )
        scope_value = (
            "clause"
            if raw_scope is None
            or (isinstance(raw_scope, float) and pd.isna(raw_scope))
            or str(raw_scope).strip() == ""
            else str(raw_scope)
        )

        model["guidelines"].append(
            {
                "key": guideline_key,
                "fallback_from_key": fallback_from_key,
                "guideline": str(guideline_text),
                "priority": priority_value,
                "scope": scope_value,
                "clause_keys": (
                    None if pd.isna(raw_clause_keys) else str(raw_clause_keys).split(",")
                ),
            }
        )

    return model


def write_tokens_usage(
    category: str,
    source_id: str,
    model: str,
    usage: dict | None,
    num_paragraphs: int,
    base_dir: Path,
) -> None:
    try:
        out_dir = base_dir / "dataset" / "output" / category
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "tokens.jsonl"
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_id": source_id,
            "provider": "openai",
            "model": model,
            "num_paragraphs": num_paragraphs,
            "usage": usage or {},
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error writing tokens usage: {e}")
        return
