"""Stage 6: item and receipt-level categories (spec §4 stage 6, interfaces §5).

Models: models/categorizer/{items,merchants}.joblib (scripts/train_categorizer.py), each a dict
{"model": sklearn pipeline, "theta": float}. Without them, the keyword dictionary below is used.
"""
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from receipt.schema import Receipt

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "categorizer"
# A confident merchant in one of these hands its category to every item (a restaurant's items are dining).
INHERIT = {"dining", "transport_fuel", "clothing", "electronics", "services"}

# Keyword baseline: generic words only (no brand names), whole word with an optional plural "s"/"es".
ITEM_KEYWORDS = {
    "groceries": "banana apple orange lemon lime avocado tomato potato onion garlic lettuce spinach carrot "
                 "pepper cucumber berry strawberry blueberry grape fruit vegetable salad milk egg cheese butter "
                 "yogurt cream bread bagel tortilla muffin rice pasta noodle spaghetti flour sugar salt oil "
                 "vinegar sauce cereal oat oatmeal granola chicken beef pork turkey ham bacon sausage fish "
                 "salmon tuna shrimp steak meat frozen pizza soup bean chip cracker cookie candy chocolate "
                 "snack nut almond peanut honey jam syrup spice seasoning organic",
    "drinks": "water soda cola juice coffee tea beer wine vodka whiskey rum tequila gin ale lager ipa "
              "lemonade kombucha seltzer sparkling espresso latte drink beverage",
    "household": "paper towel tissue toilet napkin detergent bleach cleaner trash bag foil wrap dish "
                 "sponge laundry softener battery bulb diaper wipe dog cat litter pet candle",
    "health": "shampoo conditioner soap toothpaste toothbrush floss deodorant lotion sunscreen vitamin "
              "supplement razor medicine tablet capsule pain allergy bandage cosmetic mascara lipstick",
    "dining": "burger cheeseburger fries combo meal sandwich taco burrito wrap pizza wing nugget entree "
              "appetizer dessert sundae nasi ayam goreng mie",
}
MERCHANT_KEYWORDS = {
    "groceries": "grocery groceries market supermarket foods mart produce",
    "dining": "restaurant cafe coffee pizza pizzeria grill kitchen diner burger taco bistro sushi "
              "bakery deli eatery bbq noodle thai chinese mexican italian",
    "drinks": "bar pub tavern brewery liquor wine spirits saloon taproom",
    "household": "hardware home lumber supply houseware",
    "health": "pharmacy drug drugs chemist apothecary rx",
    "transport_fuel": "gas fuel petroleum station oil",
    "clothing": "apparel clothing clothes shoe shoes boutique fashion outfitters footwear",
    "electronics": "electronics wireless mobile phone computer",
    "services": "salon barber hair laundry cleaners laundromat nails spa",
}


def _compile(table):
    return {c: re.compile(r"\b(?:" + "|".join(map(re.escape, kw.split())) + r")(?:e?s)?\b") for c, kw in table.items()}


_ITEM_RE, _MERCHANT_RE = _compile(ITEM_KEYWORDS), _compile(MERCHANT_KEYWORDS)


def keyword(name: str, table: dict, default: str) -> tuple[str, bool]:
    """(category with most keyword hits, any hit?). Ties go to the earlier category."""
    text = " ".join(re.findall(r"[a-z0-9]+", name.lower()))
    hits = {c: len(rx.findall(text)) for c, rx in table.items()}
    best = max(hits, key=hits.get)
    return (best, True) if hits[best] else (default, False)


@lru_cache(maxsize=None)
def _load(kind: str):
    p = MODEL_DIR / f"{kind}.joblib"
    if not p.exists():
        return None
    import joblib
    return joblib.load(p)


def items_category(names: list[str]) -> list[str]:
    m = _load("items")
    if m:
        return [str(c) for c in m["model"].predict(names)] if names else []
    return [keyword(n, _ITEM_RE, "groceries")[0] for n in names]


def merchant_category(name: str) -> tuple[str, bool]:
    """(category, confident). Confident = probability >= theta (tuned on val), or a keyword hit."""
    m = _load("merchants")
    if m:
        p = m["model"].predict_proba([name])[0]
        i = int(p.argmax())
        return str(m["model"].classes_[i]), bool(p[i] >= m["theta"])
    return keyword(name, _MERCHANT_RE, "other")


def apply(r: Receipt) -> Receipt:
    """Fill item categories and the receipt category (returns a copy)."""
    r = r.model_copy(deep=True)
    m_cat, sure = merchant_category(r.merchant) if r.merchant else ("other", False)
    if sure and m_cat in INHERIT:
        cats = [m_cat] * len(r.items)
    else:
        cats = items_category([it.name for it in r.items])
    for it, c in zip(r.items, cats):
        it.category = c
    if sure:
        r.category = m_cat
    else:
        money = Counter()
        for it in r.items:
            money[it.category] += abs(float(it.price or 0))
        money = +money or Counter(it.category for it in r.items)  # no prices -> item count
        r.category = money.most_common(1)[0][0] if money else "other"
    return r
