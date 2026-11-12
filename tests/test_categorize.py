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


def test_supermarket_items_are_never_dining(mode):
    # A bare dish-like name ("BANANAS") reads as a menu item to the item model; a confident
    # non-restaurant merchant rules dining out for its items.
    r = categorize.apply(Receipt(merchant="WHOLE FOODS MARKET", items=[Item(name="BANANAS", qty="1", price="1.98")]))
    assert r.items[0].category != "dining"


def test_merchant_key():
    assert categorize.merchant_key("Walmart #1234") == categorize.merchant_key("WAL MART") == "walmart"
    assert categorize.merchant_key("DOLLARGENERALSTORE#04740") == "dollargeneral"
    assert categorize.merchant_key("MR. D.I.Y. (M) SDN BHD") == "mrdiy"
    assert categorize.merchant_key("The Home Depot") == "homedepot"


def test_fast_food_without_merchant(mode):
    # No merchant: the money-weighted item vote decides dining, and the items follow it.
    r = categorize.apply(Receipt(merchant=None, items=[
        Item(name="CHEESEBURGER", price="7.99"), Item(name="FRIES", price="3.49"), Item(name="COKE", price="2.00")]))
    assert r.category == "dining"
    assert {it.category for it in r.items} == {"dining"}


@pytest.mark.parametrize("pm, want", [({}, "dining"), ({"groceries": 0.6, "dining": 0.1}, "groceries")])
def test_unsure_merchant_settles_a_split(monkeypatch, tmp_path, pm, want):
    # Keyword baseline items split evenly (dining, groceries); an unsure merchant leaning groceries decides.
    monkeypatch.setattr(categorize, "MODEL_DIR", tmp_path)
    categorize._load.cache_clear()
    monkeypatch.setattr(categorize, "_merchant", lambda name: ("groceries", False, pm))
    r = Receipt(merchant="CORNER 24", items=[Item(name="BURGER", price="5.00"), Item(name="BANANAS", price="5.00")])
    assert categorize.apply(r).category == want
    categorize._load.cache_clear()


def test_general_store_receipt_goes_by_money(mode):
    # A general store (Walmart) hands no category to its items; the receipt follows the money.
    if mode == "models":
        assert categorize.merchant_category("WALMART #1234") == ("general", True)
    r = categorize.apply(Receipt(merchant="WALMART #1234", items=[
        Item(name="BANANAS", price="1.98"), Item(name="PAPER TOWELS", price="12.99")]))
    assert r.category == r.items[1].category != "dining"
