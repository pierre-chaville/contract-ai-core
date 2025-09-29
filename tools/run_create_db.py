from __future__ import annotations

"""
Create a local SQLite test database with a single table `CONTRACTS`.

Fields:
- metadata: JSON, conforms to `contract_ai_core.schema.ContractMetadata`
- full_text: TEXT, the full contract text
- clauses: JSON, {clause_id: clause_text}
- datapoints: JSON, {datapoint_id: value}
- guidelines: JSON, {guideline_id: matched_bool}

Usage:
  python tools/run_create_db.py --db tools/test_contracts.sqlite
  python tools/run_create_db.py --db tools/test_contracts.sqlite --drop  # recreate
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field
from sqlalchemy import JSON, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Try to import the ContractMetadata Pydantic model for type validation
try:
    from contract_ai_core.schema import ContractMetadata  # type: ignore
except Exception:  # pragma: no cover - fallback if local edits break imports

    class ContractMetadata(BaseModel):  # type: ignore
        """Fallback when schema import is unavailable.

        Stores metadata as an unvalidated dictionary, so the DB layout
        remains the same (metadata JSON), but validation is skipped.
        """

        data: Dict[str, Any] = Field(default_factory=dict)


class ContractPayload(BaseModel):
    """Pydantic model describing what a `CONTRACTS` row represents."""

    metadata: ContractMetadata | Dict[str, Any]
    full_text: str
    clauses: Dict[str, str] = Field(default_factory=dict)
    datapoints: Dict[str, Any] = Field(default_factory=dict)
    guidelines: Dict[str, bool] = Field(default_factory=dict)


class Base(DeclarativeBase):
    pass


class ContractRecord(Base):
    __tablename__ = "CONTRACTS"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Expanded metadata columns (from ContractMetadata)
    contract_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True, unique=True, index=True
    )
    contract_number: Mapped[str | None] = mapped_column(String(255), nullable=True)
    contract_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    contract_type_version: Mapped[str | None] = mapped_column(String(255), nullable=True)
    contract_date: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_amendment_date: Mapped[str | None] = mapped_column(String(64), nullable=True)
    number_amendments: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    party_name_1: Mapped[str | None] = mapped_column(String(255), nullable=True)
    party_role_1: Mapped[str | None] = mapped_column(String(255), nullable=True)
    party_name_2: Mapped[str | None] = mapped_column(String(255), nullable=True)
    party_role_2: Mapped[str | None] = mapped_column(String(255), nullable=True)
    department: Mapped[str | None] = mapped_column(String(255), nullable=True)
    contract_owner: Mapped[str | None] = mapped_column(String(255), nullable=True)
    business_purpose: Mapped[str | None] = mapped_column(Text, nullable=True)
    full_text: Mapped[str] = mapped_column(Text, nullable=False)
    clauses: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    datapoints: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    guidelines: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    def __repr__(self) -> str:  # pragma: no cover - representation helper
        return (
            f"<ContractRecord id={self.id} contract_id={self.contract_id!r} "
            f"clauses={len(self.clauses)} datapoints={len(self.datapoints)} "
            f"guidelines={len(self.guidelines)}>"
        )


def _ensure_parent_directory(db_path: str) -> None:
    abs_path = os.path.abspath(db_path)
    parent = os.path.dirname(abs_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _sqlite_url(db_path: str) -> str:
    # Use absolute path to avoid confusion with working directory
    abs_path = os.path.abspath(db_path)
    return f"sqlite:///{abs_path}"


def create_database(drop_first: bool = False, echo: bool = False) -> None:
    """Create (and optionally recreate) the SQLite database and `CONTRACTS` table."""

    repo_root = Path(__file__).resolve().parents[1]
    db_path = os.path.join(repo_root, "dataset", "contracts.sqlite")
    print(db_path)
    _ensure_parent_directory(db_path)
    engine = create_engine(_sqlite_url(db_path), echo=echo, future=True)

    if drop_first:
        Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a test database with CONTRACTS table.")
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop existing tables before creating them again.",
    )
    parser.add_argument(
        "--echo",
        action="store_true",
        help="Enable SQLAlchemy engine echo for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_database(drop_first=args.drop, echo=args.echo)


if __name__ == "__main__":
    main()
