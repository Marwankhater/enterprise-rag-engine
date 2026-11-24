"""Read-only Text-to-SQL over the sales schema.

The model writes SQL; nothing it writes is trusted. Four independent layers stop a bad query:
  1. sqlglot parses it and only a single SELECT (or WITH ... SELECT) over allow-listed tables passes,
     with dangerous functions, SELECT INTO and row locks rejected, and a LIMIT forced on.
  2. It runs as a database role that can only SELECT from the sales schema.
  3. The role defaults to read-only transactions, and the session sets READ ONLY explicitly too.
  4. A statement timeout bounds how long any query can run.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import sqlglot
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlglot import exp

from app.agents.llm import LLM, LLMError
from app.core.config import AccessLevel, Principal, Settings
from app.core.observability import span

log = logging.getLogger(__name__)


class UnsafeQuery(ValueError):
    pass


FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, name)
    for name in (
        "Insert",
        "Update",
        "Delete",
        "Drop",
        "Create",
        "Alter",
        "AlterTable",
        "Merge",
        "TruncateTable",
        "Command",
        "Grant",
        "Revoke",
        "Copy",
        "Into",
        "Lock",
        "Set",
        "Use",
        "Transaction",
        "Commit",
        "Rollback",
    )
    if hasattr(exp, name)
)
FORBIDDEN_FUNCTIONS = frozenset(
    {
        "pg_sleep",
        "pg_sleep_for",
        "pg_sleep_until",
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_stat_file",
        "lo_import",
        "lo_export",
        "lo_get",
        "dblink",
        "dblink_exec",
        "set_config",
        "current_setting",
        "pg_terminate_backend",
        "pg_cancel_backend",
        "pg_reload_conf",
        "query_to_xml",
        "txid_current",
        "pg_advisory_lock",
        "pg_notify",
    }
)


def validate_sql(sql: str, *, schema: str, allowed_tables: list[str], max_rows: int) -> str:
    """Return a safe, LIMIT-bounded version of `sql` or raise UnsafeQuery with the reason."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeQuery("empty query")
    try:
        statements = sqlglot.parse(cleaned, read="postgres")
    except sqlglot.errors.ParseError as exc:
        raise UnsafeQuery(f"could not parse SQL: {exc}") from exc
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise UnsafeQuery("exactly one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        raise UnsafeQuery(f"only SELECT queries are allowed, got {type(tree).__name__}")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise UnsafeQuery(f"{type(node).__name__} is not allowed")
        if isinstance(node, exp.Func):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if name in FORBIDDEN_FUNCTIONS or name.startswith("pg_"):
                raise UnsafeQuery(f"function {name} is not allowed")
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    allowed = {t.lower() for t in allowed_tables}
    for table in tree.find_all(exp.Table):
        name, db = table.name.lower(), (table.db or "").lower()
        if not name:
            raise UnsafeQuery("table functions are not allowed")
        if not db and name in cte_names:
            continue
        if db not in ("", schema.lower()) or table.catalog:
            raise UnsafeQuery(f"schema {db or table.catalog} is not allowed")
        if name not in allowed:
            raise UnsafeQuery(f"table {name} is not allowed")
        table.set("db", exp.to_identifier(schema))
    limit = tree.args.get("limit")
    current = None
    if isinstance(limit, exp.Limit) and isinstance(limit.expression, exp.Literal) and limit.expression.is_int:
        current = int(limit.expression.this)
    if current is None or current > max_rows:
        tree = tree.limit(max_rows)
    return tree.sql(dialect="postgres")


SCHEMA_DOC = """Schema `sales` (PostgreSQL). Monetary values are USD.
customers(id, name, country, region 'US'|'EU', plan 'Starter'|'Pro'|'Enterprise', seats, created_at date)
products(id, sku, name, category, unit_price)
orders(id, customer_id -> customers.id, order_date date, billing 'monthly'|'annual',
       status 'paid'|'refunded'|'cancelled', total)
order_items(id, order_id -> orders.id, product_id -> products.id, quantity, unit_price)
Revenue means the sum of orders.total where status = 'paid'."""

SQL_PROMPT = """You write one PostgreSQL SELECT query that answers the question from this schema.
{schema}
Rules: read-only SELECT only, qualify tables with sales., no comments, at most {max_rows} rows,
use ISO dates. Reply with a JSON object: {{"sql": "...", "explanation": "one sentence"}}.
If the schema cannot answer the question, reply {{"sql": null, "explanation": "why"}}."""


@dataclass
class SQLResult:
    question: str
    sql: str | None
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    truncated: bool = False
    error: str | None = None

    def as_context(self, max_rows: int = 50) -> str:
        if self.error or not self.sql:
            return f"SQL could not answer this: {self.error or 'not covered by the sales schema'}"
        header = " | ".join(self.columns)
        body = "\n".join(" | ".join(_fmt(v) for v in row) for row in self.rows[:max_rows])
        more = f"\n(showing {max_rows} of {len(self.rows)} rows)" if len(self.rows) > max_rows else ""
        return f"Query: {self.sql}\n{header}\n{body}{more}"


def _fmt(value: Any) -> str:
    if isinstance(value, Decimal):
        return f"{value:.2f}"
    return "" if value is None else str(value)


class SQLTool:
    MIN_CLEARANCE = AccessLevel.INTERNAL

    def __init__(self, llm: LLM, settings: Settings, engine: AsyncEngine | None = None) -> None:
        self.llm = llm
        self.settings = settings
        url = settings.sql_tool_database_url or settings.database_url
        self.engine = engine or create_async_engine(url, pool_size=3, max_overflow=2, pool_pre_ping=True)

    def allowed_for(self, principal: Principal) -> bool:
        return principal.clearance >= self.MIN_CLEARANCE

    async def generate(self, question: str, error: str | None = None, previous: str | None = None) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": SQL_PROMPT.format(schema=SCHEMA_DOC, max_rows=self.settings.sql_max_rows)},
            {"role": "user", "content": question},
        ]
        if error and previous:
            messages.append({"role": "assistant", "content": previous})
            messages.append({"role": "user", "content": f"That query was rejected: {error}. Write a corrected one."})
        result = await self.llm.chat(messages, json_mode=True, name="sql.generate", max_tokens=600)
        return result.json()

    async def execute(self, sql: str) -> tuple[list[str], list[list[Any]]]:
        async with self.engine.connect() as conn:
            await conn.execute(text("set transaction read only"))
            await conn.execute(text(f"set local statement_timeout = {int(self.settings.sql_timeout_ms)}"))
            result = await conn.execute(text(sql))
            columns = list(result.keys())
            rows = [list(r) for r in result.fetchall()]
            await conn.rollback()
        return columns, rows

    async def run(self, question: str, principal: Principal) -> SQLResult:
        if not self.allowed_for(principal):
            return SQLResult(question, None, error="your access level does not include sales data")
        with span("sql.tool") as current:
            error: str | None = None
            previous: str | None = None
            for _attempt in range(2):
                try:
                    reply = await self.generate(question, error, previous)
                except LLMError as exc:
                    return SQLResult(question, None, error=str(exc))
                raw = reply.get("sql")
                if not raw:
                    return SQLResult(question, None, error=reply.get("explanation") or "not answerable from sales data")
                previous = str(raw)
                try:
                    safe = validate_sql(
                        previous,
                        schema=self.settings.sql_schema,
                        allowed_tables=self.settings.sql_allowed_tables,
                        max_rows=self.settings.sql_max_rows,
                    )
                except UnsafeQuery as exc:
                    error = str(exc)
                    log.warning("sql rejected", extra={"reason": error, "sql": previous[:500]})
                    continue
                current.set_attribute("sql", safe)
                try:
                    columns, rows = await self.execute(safe)
                except Exception as exc:
                    error = re.sub(r"\s+", " ", str(getattr(exc, "orig", exc)))[:300]
                    continue
                return SQLResult(question, safe, columns, rows, truncated=len(rows) >= self.settings.sql_max_rows)
            return SQLResult(question, previous, error=error)
