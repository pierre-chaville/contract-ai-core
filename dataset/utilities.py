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

    model = json.loads(open(json_path, encoding="utf-8").read())
    model["clauses"] = []
    model["enums"] = []
    model["datapoints"] = []

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
