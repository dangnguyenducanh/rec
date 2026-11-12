"""categorize.apply on a synthetic grocery receipt and a synthetic restaurant receipt,
with the trained models (when present) and with the keyword fallback."""
import pytest

from receipt import categorize
from receipt.schema import Item, Receipt

GROCERY = Receipt(merchant="SAFEWAY", total="13.39", items=[
    Item(name="ORGANIC BANANAS", price="1.98"), Item(name="WHOLE MILK 1GAL", price="4.29"),
    Item(name="SOURDOUGH BREAD", price="3.99"), Item(name="LARGE BROWN EGGS 12CT", price="3.13")])
RESTAURANT = Receipt(merchant="Luigi's Italian Restaurant", total="41.50", items=[
    Item(name="CHICKEN PARMESAN", price="18.50"), Item(name="LASAGNA", price="17.00"),
    Item(name="ICED TEA", price="3.00"), Item(name="TIRAMISU", price="3.00")])


@pytest.fixture(params=["models", "keywords"])
def mode(request, monkeypatch, tmp_path):
    if request.param == "models":
        if not (categorize.MODEL_DIR / "items.joblib").exists():
            pytest.skip("no trained categorizer")
    else:
        monkeypatch.setattr(categorize, "MODEL_DIR", tmp_path)  # empty dir -> keyword fallback
    categorize._load.cache_clear()
    yield request.param
    categorize._load.cache_clear()


def test_grocery(mode):
    r = categorize.apply(GROCERY)
    assert r.category == "groceries"
    assert [it.category for it in r.items].count("groceries") >= 3
    assert GROCERY.category == "other"  # input left untouched


def test_restaurant(mode):
    r = categorize.apply(RESTAURANT)
    assert r.category == "dining"
    assert {it.category for it in r.items} == {"dining"}  # items inherit a confident dining merchant
