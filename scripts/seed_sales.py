"""Deterministic sample data for the `sales` schema (fictional Acme Cloud customers and orders).

All names and numbers are generated sample content. The same seed always produces the same rows, so
the SQL checks in tests/golden_dataset.json stay valid after a re-seed.

    python -m scripts.seed_sales            # uses DATABASE_URL, or the local default below
    python -m scripts.seed_sales --if-empty # only when the sales tables are empty

Pricing follows examples/product-faq.txt: Starter is free, Pro is $29 per user per month, Enterprise
is custom-priced (list reference $79 per user per month, with a negotiated per-customer discount), and
annual billing costs 10 times the monthly price ("annual billing saves two months").
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

import psycopg

DEFAULT_URL = "postgresql://rag:rag@localhost:5432/rag"
SEED = 42
N_CUSTOMERS = 120
N_ORDERS = 1500
FIRST_ORDER_DAY = date(2025, 1, 1)
LAST_ORDER_DAY = date(2026, 10, 31)
CENT = Decimal("0.01")

# sku, name, category, monthly list price (one-time price for services)
PRODUCTS: list[tuple[str, str, str, Decimal]] = [
    ("PLAN-STARTER", "Starter plan", "plan", Decimal("0.00")),
    ("PLAN-PRO", "Pro plan", "plan", Decimal("29.00")),
    ("PLAN-ENTERPRISE", "Enterprise plan", "plan", Decimal("79.00")),
    ("ADDON-STORAGE-100GB", "Extra storage (100 GB block)", "add-on", Decimal("10.00")),
    ("ADDON-PREMIUM-SUPPORT", "Premium support (24/7 phone)", "add-on", Decimal("199.00")),
    ("ADDON-SSO", "Single sign-on (SSO) for Pro", "add-on", Decimal("6.00")),
    ("ADDON-AUDIT-LOG", "Extended audit log retention", "add-on", Decimal("49.00")),
    ("SVC-ONBOARDING", "Guided onboarding package", "service", Decimal("1500.00")),
]
SKU_ID = {sku: i for i, (sku, *_rest) in enumerate(PRODUCTS, start=1)}
LIST_PRICE = {sku: price for sku, _n, _c, price in PRODUCTS}

US_COUNTRIES = [("United States", "Inc.", 5), ("Canada", "Ltd.", 1)]
EU_COUNTRIES = [
    ("Germany", "GmbH", 3),
    ("France", "SAS", 2),
    ("Netherlands", "B.V.", 2),
    ("Spain", "S.L.", 1),
    ("Italy", "S.r.l.", 1),
    ("Ireland", "Ltd", 1),
    ("Sweden", "AB", 1),
    ("Poland", "Sp. z o.o.", 1),
    ("Belgium", "NV", 1),
    ("Austria", "GmbH", 1),
]
NAME_FIRST = [
    "Alder",
    "Birchwood",
    "Cobalt",
    "Driftline",
    "Ember",
    "Fjord",
    "Granite",
    "Harbor",
    "Ironleaf",
    "Juniper",
    "Kestrel",
    "Lumen",
    "Meridian",
    "Northgate",
    "Oakridge",
    "Pinecrest",
    "Quarry",
    "Redwood",
    "Saltmarsh",
    "Tidewater",
    "Upland",
    "Vantage",
    "Willow",
    "Yarrow",
    "Zephyr",
]
NAME_SECOND = [
    "Analytics",
    "Logistics",
    "Health",
    "Foods",
    "Systems",
    "Labs",
    "Retail",
    "Energy",
    "Media",
    "Robotics",
    "Finance",
    "Studios",
]


@dataclass
class Customer:
    id: int
    name: str
    country: str
    region: str
    plan: str
    seats: int
    created_at: date
    billing: str
    discount: Decimal  # negotiated discount on the Enterprise seat price


def database_url() -> str:
    url = os.environ.get("DATABASE_URL", DEFAULT_URL)
    for driver in ("+asyncpg", "+psycopg2", "+psycopg"):
        url = url.replace(f"postgresql{driver}://", "postgresql://")
    return url


def money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def weighted(rng: random.Random, options: list[tuple[str, str, int]]) -> tuple[str, str]:
    pool = [(country, suffix) for country, suffix, weight in options for _ in range(weight)]
    return rng.choice(pool)


def make_customers(rng: random.Random) -> list[Customer]:
    names = [f"{a} {b}" for a in NAME_FIRST for b in NAME_SECOND]
    rng.shuffle(names)
    customers: list[Customer] = []
    for i in range(1, N_CUSTOMERS + 1):
        region = "US" if rng.random() < 0.55 else "EU"
        country, suffix = weighted(rng, US_COUNTRIES if region == "US" else EU_COUNTRIES)
        roll = rng.random()
        plan = "Starter" if roll < 0.25 else "Pro" if roll < 0.80 else "Enterprise"
        if plan == "Starter":
            seats, billing = rng.randint(1, 3), "monthly"
        elif plan == "Pro":
            seats, billing = rng.randint(3, 60), "annual" if rng.random() < 0.3 else "monthly"
        else:
            seats, billing = rng.randint(50, 600), "annual" if rng.random() < 0.7 else "monthly"
        discount = Decimal(rng.choice(["0", "0.05", "0.10", "0.15", "0.20"])) if plan == "Enterprise" else Decimal(0)
        created = date(2024, 3, 1) + timedelta(days=rng.randrange((date(2026, 9, 30) - date(2024, 3, 1)).days + 1))
        customers.append(
            Customer(i, f"{names[i - 1]} {suffix}", country, region, plan, seats, created, billing, discount)
        )
    return customers


def line_price(sku: str, billing: str, discount: Decimal = Decimal(0)) -> Decimal:
    price = LIST_PRICE[sku] * (1 - discount)
    if billing == "annual" and not sku.startswith("SVC-"):
        price *= 10  # annual billing saves two months
    return money(price)


def make_orders(rng: random.Random, customers: list[Customer]) -> tuple[list[tuple], list[tuple]]:
    orders: list[tuple] = []
    items: list[tuple] = []
    for order_id in range(1, N_ORDERS + 1):
        c = rng.choice(customers)
        start = max(c.created_at, FIRST_ORDER_DAY)
        order_date = start + timedelta(days=rng.randrange((LAST_ORDER_DAY - start).days + 1))
        roll = rng.random()
        status = "refunded" if roll < 0.05 else "cancelled" if roll < 0.08 else "paid"

        lines: list[tuple[str, int, Decimal]] = []
        plan_sku = {"Starter": "PLAN-STARTER", "Pro": "PLAN-PRO", "Enterprise": "PLAN-ENTERPRISE"}[c.plan]
        lines.append((plan_sku, c.seats, line_price(plan_sku, c.billing, c.discount)))
        if rng.random() < 0.30:
            lines.append(("ADDON-STORAGE-100GB", rng.randint(1, 5), line_price("ADDON-STORAGE-100GB", c.billing)))
        if c.plan == "Pro" and rng.random() < 0.15:
            lines.append(("ADDON-SSO", c.seats, line_price("ADDON-SSO", c.billing)))
        if c.plan == "Pro" and rng.random() < 0.10:
            lines.append(("ADDON-PREMIUM-SUPPORT", 1, line_price("ADDON-PREMIUM-SUPPORT", c.billing)))
        if c.plan == "Enterprise" and rng.random() < 0.25:
            lines.append(("ADDON-AUDIT-LOG", 1, line_price("ADDON-AUDIT-LOG", c.billing)))
        if c.plan != "Starter" and rng.random() < 0.04:
            lines.append(("SVC-ONBOARDING", 1, line_price("SVC-ONBOARDING", c.billing)))

        total = money(sum((price * qty for _sku, qty, price in lines), Decimal(0)))
        orders.append((order_id, c.id, order_date, c.billing, status, total))
        items.extend((order_id, SKU_ID[sku], qty, price) for sku, qty, price in lines)
    return orders, items


def seed(url: str | None = None) -> dict[str, int]:
    rng = random.Random(SEED)
    customers = make_customers(rng)
    orders, items = make_orders(rng, customers)
    with psycopg.connect(url or database_url()) as conn, conn.cursor() as cur:
        cur.execute(
            "truncate sales.order_items, sales.orders, sales.customers, sales.products restart identity cascade"
        )
        cur.executemany(
            "insert into sales.products (id, sku, name, category, unit_price) values (%s, %s, %s, %s, %s)",
            [(SKU_ID[sku], sku, name, cat, price) for sku, name, cat, price in PRODUCTS],
        )
        cur.executemany(
            "insert into sales.customers (id, name, country, region, plan, seats, created_at) "
            "values (%s, %s, %s, %s, %s, %s, %s)",
            [(c.id, c.name, c.country, c.region, c.plan, c.seats, c.created_at) for c in customers],
        )
        cur.executemany(
            "insert into sales.orders (id, customer_id, order_date, billing, status, total) "
            "values (%s, %s, %s, %s, %s, %s)",
            orders,
        )
        cur.executemany(
            "insert into sales.order_items (order_id, product_id, quantity, unit_price) values (%s, %s, %s, %s)", items
        )
        for table in ("products", "customers", "orders", "order_items"):
            cur.execute(
                f"select setval(pg_get_serial_sequence('sales.{table}', 'id'), "
                f"(select coalesce(max(id), 1) from sales.{table}))"
            )
    return {"products": len(PRODUCTS), "customers": len(customers), "orders": len(orders), "order_items": len(items)}


def already_seeded() -> bool:
    with psycopg.connect(database_url()) as conn, conn.cursor() as cur:
        cur.execute("select exists (select 1 from sales.orders)")
        return bool(cur.fetchone()[0])


def main() -> None:
    if "--if-empty" in sys.argv[1:] and already_seeded():
        print("sales data already present, skipping")
        return
    counts = seed()
    print(", ".join(f"{table}: {n}" for table, n in counts.items()))


if __name__ == "__main__":
    main()
