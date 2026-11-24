"""The Text-to-SQL guard rails: the validator, and the read-only role behind it."""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings
from app.retrieval.sql_tool import SQLTool, UnsafeQuery, validate_sql
from tests.conftest import ScriptedLLM, principal

RULES = {"schema": "sales", "allowed_tables": ["products", "customers", "orders", "order_items"], "max_rows": 200}


@pytest.mark.parametrize(
    "sql",
    [
        "select count(*) from orders where status = 'refunded'",
        "SELECT c.name, sum(o.total) FROM sales.orders o JOIN sales.customers c ON c.id = o.customer_id GROUP BY 1",
        "with t as (select customer_id, sum(total) s from orders group by 1) select * from t order by s desc limit 5",
        "select region, count(*) from customers group by region union all select 'all', count(*) from customers",
    ],
)
def test_reads_over_allowed_tables_pass(sql: str) -> None:
    safe = validate_sql(sql, **RULES)
    assert "LIMIT" in safe.upper()
    assert "sales." in safe


def test_limit_is_forced_and_capped() -> None:
    assert validate_sql("select * from orders", **RULES).endswith("LIMIT 200")
    assert validate_sql("select * from orders limit 100000", **RULES).endswith("LIMIT 200")
    assert validate_sql("select * from orders limit 5", **RULES).endswith("LIMIT 5")


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("delete from orders", "only SELECT"),
        ("update orders set total = 0", "only SELECT"),
        ("insert into orders values (1)", "only SELECT"),
        ("drop table orders", "only SELECT"),
        ("truncate sales.orders", "only SELECT"),
        ("alter table orders add column x int", "only SELECT"),
        ("select 1; drop table orders", "exactly one statement"),
        ("select * into backup from orders", "Into"),
        ("select * from orders for update", "Lock"),
        ("select pg_sleep(30)", "pg_sleep"),
        ("select pg_read_file('/etc/passwd')", "pg_read_file"),
        ("select set_config('role', 'rag', false)", "set_config"),
        ("select * from public.documents", "schema public"),
        ("select * from child_chunks", "table child_chunks"),
        ("select * from pg_catalog.pg_authid", "schema pg_catalog"),
        ("select * from information_schema.columns", "schema information_schema"),
        ("select * from generate_series(1, 1000000000)", "table functions"),
        ("copy orders to '/tmp/out.csv'", "only SELECT"),
        ("", "empty"),
    ],
)
def test_dangerous_queries_are_rejected(sql: str, reason: str) -> None:
    with pytest.raises(UnsafeQuery, match=reason):
        validate_sql(sql, **RULES)


async def test_reader_role_cannot_write_even_without_the_validator(migrated: None) -> None:
    url = get_settings().database_url.replace("rag:rag@", "rag_reader:rag_reader@")
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            assert (await conn.execute(text("select count(*) from sales.orders"))).scalar_one() >= 0
            with pytest.raises(Exception, match=r"read-only|permission denied"):
                await conn.execute(text("delete from sales.orders"))
        async with engine.connect() as conn:
            with pytest.raises(Exception, match="permission denied"):
                await conn.execute(text("select * from public.documents"))
    finally:
        await engine.dispose()


async def test_tool_answers_and_respects_clearance(migrated: None) -> None:
    settings = get_settings()
    tool = SQLTool(ScriptedLLM(), settings)
    try:
        result = await tool.run("How many refunded orders are there?", principal("e", clearance="internal"))
        assert result.error is None and result.columns == ["refunded_orders"]
        denied = await tool.run("How many refunded orders are there?", principal("v", clearance="public"))
        assert denied.sql is None and "access" in (denied.error or "")
    finally:
        await tool.engine.dispose()


async def test_rejected_query_gets_one_repair_attempt(migrated: None) -> None:
    llm = ScriptedLLM(
        {"sql.generate": ['{"sql": "delete from sales.orders"}', '{"sql": "select count(*) as n from sales.orders"}']}
    )
    tool = SQLTool(llm, get_settings())
    try:
        result = await tool.run("how many orders", principal("e", clearance="internal"))
        assert result.columns == ["n"]
        repair_prompt = llm.calls[-1][1][-1]["content"]
        assert "rejected" in repair_prompt
    finally:
        await tool.engine.dispose()
