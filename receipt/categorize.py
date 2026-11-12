"""Stage 6: item and receipt-level categories (spec §4 stage 6, interfaces §5).

Models: models/categorizer/{items,merchants}.joblib (scripts/train_categorizer.py), each a dict
{"model": sklearn pipeline, "theta": float}; merchants also has "theta_class" {category: stricter theta} and
"lookup" {merchant_key: category}, an exact-name table over every labelled merchant name, which may say "general"
(department/variety store). Without the models, the keyword dictionary below is used.
"""
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from receipt.schema import Receipt

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "categorizer"
# A confident merchant in one of these hands its category to every item (a restaurant's items are dining).
INHERIT = {"dining", "transport_fuel", "clothing", "electronics", "services"}
MERCHANT_SHARE = 0.3  # an unsure merchant's probabilities still count for this share of the receipt vote

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


LEGAL = re.compile(r"\b(?:sdn bhd|s b|bhd|inc|llc|ltd|limited|corp|co|plc|pte)\b")


def squash(s: str) -> str:
    """Lowercase letters and digits only: 'CD PORK CHOP' and 'CDPORKCHOP' read the same (a model preprocessor)."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def merchant_key(s: str) -> str:
    """Strict lookup key: 'Walmart #1234' / 'WAL MART' -> 'walmart', 'MR. D.I.Y. (M) SDN BHD' -> 'mrdiy'."""
    s = s.lower().replace("&", " and ")
    s = re.sub(r"\(.*?\)|#\s*\d+|\b\d{3,}\b|\d+\W*$", " ", s)  # (M), store numbers, trailing digits
    s = re.sub(r"\b([a-z])\.\s*(?=[a-z]\b)", r"\1", s)  # d. i. y. -> diy
    s = re.sub(r"[^a-z0-9]", "", LEGAL.sub(" ", " ".join(re.findall(r"[a-z0-9]+", s))))  # no spaces: gold text drops them
    s = re.sub(r"^(?:welcometo|the)(?=[a-z0-9]{3})", "", s)
    return re.sub(r"(?<=[a-z0-9]{3})(?:stores?|restaurants?|shop)$", "", s)


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


def items_category(names: list[str], exclude: frozenset = frozenset()) -> list[str]:
    """Category per item name; `exclude` drops categories the receipt rules out (the best remaining wins)."""
    m = _load("items")
    if m and names:
        classes = [str(c) for c in m["model"].classes_]
        keep = [i for i, c in enumerate(classes) if c not in exclude] or list(range(len(classes)))
        return [classes[max(keep, key=lambda i: p[i])] for p in m["model"].predict_proba(names)]
    if m:
        return []
    cats = [keyword(n, _ITEM_RE, "groceries")[0] for n in names]
    return ["groceries" if c in exclude else c for c in cats]


def _merchant(name: str) -> tuple[str, bool, dict]:
    """(category, confident, class probabilities). Confident = an exact lookup hit, probability >= the category's
    theta (tuned on val; stricter for categories that overwrite every item), or a keyword hit."""
    m = _load("merchants")
    if m:
        if hit := m.get("lookup", {}).get(merchant_key(name)):
            return hit, True, {}
        p = m["model"].predict_proba([name])[0]
        classes = [str(c) for c in m["model"].classes_]
        c = classes[int(p.argmax())]
        return c, bool(p.max() >= m.get("theta_class", {}).get(c, m["theta"])), dict(zip(classes, p.tolist()))
    return (*keyword(name, _MERCHANT_RE, "other"), {})


def merchant_category(name: str) -> tuple[str, bool]:
    """(category, confident): see _merchant."""
    return _merchant(name)[:2]


def _vote(names: list[str], prices: list[float], merchant_p: dict, dining: bool = True) -> str:
    """Receipt category from its items: the money-weighted mean of the item model's probabilities (item count when
    no prices); an unsure merchant's probabilities take MERCHANT_SHARE of the vote. dining=False rules dining out
    (a confident general store). Without models: most money by keyword."""
    w = np.abs(np.asarray(prices, float))
    w = w / w.sum() if w.sum() else np.full(len(names), 1 / len(names))
    m = _load("items")
    if m:
        classes = [str(c) for c in m["model"].classes_]
        vote = Counter(dict(zip(classes, m["model"].predict_proba(names).T @ w)))
    else:
        vote = Counter()
        for c, x in zip(items_category(names), w):
            vote[c] += x
    if merchant_p:
        vote = Counter({c: (1 - MERCHANT_SHARE) * vote[c] + MERCHANT_SHARE * merchant_p.get(c, 0.0)
                        for c in set(vote) | set(merchant_p)})
    if not dining:
        vote.pop("dining", None)
    return vote.most_common(1)[0][0]


def apply(r: Receipt) -> Receipt:
    """Receipt category first (a confident merchant, else the item vote), then item categories from it (returns a
    copy): an INHERIT receipt hands its category to every item (a restaurant's items are dining); any other known
    receipt rules dining out for its items, where the item model alone reads a bare "BANANAS" as a CORD dish."""
    r = r.model_copy(deep=True)
    m_cat, sure, m_p = _merchant(r.merchant) if r.merchant else ("other", False, {})
    names = [it.name for it in r.items]
    if sure and m_cat != "general":
        r.category = m_cat
    elif names:  # a general store's receipt goes by where the money went, never to dining
        r.category = _vote(names, [float(it.price or 0) for it in r.items], {} if sure else m_p, dining=not sure)
    else:
        r.category = "other"
    if r.category in INHERIT:
        cats = [r.category] * len(names)
    else:
        cats = items_category(names, frozenset({"dining"}) if r.category != "other" else frozenset())
    for it, c in zip(r.items, cats):
        it.category = c
    return r
