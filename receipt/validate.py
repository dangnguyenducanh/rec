"""Stage 7: arithmetic/date checks -> needs_review."""
from decimal import Decimal

from receipt.schema import Receipt

TOL = Decimal("0.01")


def _d(v: str | None) -> Decimal:
    return Decimal(v) if v is not None else Decimal(0)


def check(r: Receipt) -> Receipt:
    reasons = []
    if r.total is None:
        reasons.append("total is missing")
    if r.date is None:
        reasons.append("date is missing")
    prices = [_d(i.price) for i in r.items if i.price is not None]
    if r.subtotal is not None and prices and abs(sum(prices) - _d(r.subtotal)) > TOL:
        reasons.append(f"items add up to {sum(prices):.2f} but subtotal says {r.subtotal}")
    if r.subtotal is not None and r.total is not None:
        expected = _d(r.subtotal) + _d(r.tax) + _d(r.tips)
        if abs(expected - _d(r.total)) > TOL:
            reasons.append(f"subtotal + tax + tips = {expected:.2f} but total says {r.total}")
    return r.model_copy(update={"needs_review": bool(reasons), "review_reasons": reasons})
