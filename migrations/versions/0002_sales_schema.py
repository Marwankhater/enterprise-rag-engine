"""Sales schema for the Text-to-SQL tool, and a login role that can only read it.

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

READER = "rag_reader"


def upgrade() -> None:
    password = os.environ.get("SQL_READER_PASSWORD")
    if not password:
        if os.environ.get("ENVIRONMENT", "development") == "production":
            raise RuntimeError("SQL_READER_PASSWORD must be set in production")
        password = "rag_reader"
    op.execute("create schema if not exists sales")
    op.execute("""
        create table sales.customers (
            id serial primary key,
            name text not null,
            country text not null,
            region text not null check (region in ('US', 'EU')),
            plan text not null check (plan in ('Starter', 'Pro', 'Enterprise')),
            seats integer not null check (seats > 0),
            created_at date not null
        )""")
    op.execute("""
        create table sales.products (
            id serial primary key,
            sku text not null unique,
            name text not null,
            category text not null,
            unit_price numeric(10, 2) not null
        )""")
    op.execute("""
        create table sales.orders (
            id serial primary key,
            customer_id integer not null references sales.customers(id),
            order_date date not null,
            billing text not null check (billing in ('monthly', 'annual')),
            status text not null check (status in ('paid', 'refunded', 'cancelled')),
            total numeric(12, 2) not null
        )""")
    op.execute("""
        create table sales.order_items (
            id serial primary key,
            order_id integer not null references sales.orders(id) on delete cascade,
            product_id integer not null references sales.products(id),
            quantity integer not null check (quantity > 0),
            unit_price numeric(10, 2) not null
        )""")
    op.execute("create index ix_orders_customer on sales.orders(customer_id)")
    op.execute("create index ix_orders_date on sales.orders(order_date)")
    op.execute("create index ix_order_items_order on sales.order_items(order_id)")

    op.execute(f"""
        do $$
        begin
            if not exists (select 1 from pg_roles where rolname = '{READER}') then
                create role {READER} login;
            end if;
        end $$""")
    op.execute(f"alter role {READER} with login password '{password.replace(chr(39), chr(39) * 2)}'")
    op.execute(f"alter role {READER} set default_transaction_read_only = on")
    op.execute(f"alter role {READER} set statement_timeout = '5s'")
    op.execute(f"grant usage on schema sales to {READER}")
    op.execute(f"grant select on all tables in schema sales to {READER}")
    op.execute(f"revoke all on schema public from {READER}")


def downgrade() -> None:
    op.execute(f"revoke all on all tables in schema sales from {READER}")
    op.execute(f"revoke usage on schema sales from {READER}")
    op.execute("drop schema sales cascade")
    op.execute(f"drop role if exists {READER}")
