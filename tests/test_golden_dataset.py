"""The golden dataset stays consistent with the corpus, the access rules and the seeded sales data."""

from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal
from functools import cache

import pytest

from app.core.config import AccessLevel
from app.ingestion.parser import parse
from tests.conftest import EXAMPLES, ROOT

DATA = json.loads((ROOT / "tests" / "golden_dataset.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((EXAMPLES / "manifest.json").read_text(encoding="utf-8"))
CASES = DATA["cases"]


@cache
def text_of(name: str) -> str:
    return parse(EXAMPLES / name).text


def can_read(principal: str, source: str) -> bool:
    who, rule = DATA["principals"][principal], MANIFEST[source]
    level_ok = AccessLevel[rule["access_level"].upper()] <= AccessLevel[who["clearance"].upper()]
    return level_ok and (rule["department_id"] is None or rule["department_id"] in who["departments"])


def test_size_and_coverage() -> None:
    counts = Counter(c["category"] for c in CASES)
    assert len(CASES) >= 50
    for category in ("single_hop", "multi_hop", "unanswerable", "rbac_restricted", "sql", "hybrid", "arabic"):
        assert counts[category] >= 3, category
    assert len({c["id"] for c in CASES}) == len(CASES)


def test_manifest_covers_every_example_file() -> None:
    assert set(MANIFEST) == {p.name for p in EXAMPLES.iterdir() if p.name != "manifest.json"}


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_case_is_consistent(case: dict) -> None:
    for source in case["expected_sources"] + case["forbidden_sources"]:
        assert source in MANIFEST
    if case["expect"] == "answer":
        assert all(can_read(case["principal"], s) for s in case["expected_sources"])
        if case["category"] != "sql":
            assert case["reference_contexts"], "answer cases need a passage to find"
    for source in case["forbidden_sources"]:
        assert not can_read(case["principal"], source)
    if case["category"] == "rbac_restricted":
        assert case["expect"] == "refuse" and case["forbidden_sources"]
    for phrase in case["reference_contexts"]:
        assert any(phrase in text_of(s) for s in case["expected_sources"]), phrase
    if case["category"] in ("sql", "hybrid"):
        assert case["sql_check"]


async def test_sql_checks_match_the_seeded_data(migrated: None) -> None:
    import psycopg

    from app.core.config import get_settings
    from scripts.seed_sales import seed

    seed()
    url = get_settings().database_url.replace("+asyncpg", "")
    with psycopg.connect(url) as conn:
        for case in CASES:
            check = case["sql_check"]
            if not check:
                continue
            got = conn.execute(check["query"]).fetchone()[0]
            want = check["value"]
            if isinstance(want, (int, float)):
                assert Decimal(str(got)) == Decimal(str(want)), case["id"]
            else:
                assert got == want, case["id"]
