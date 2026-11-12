"""extract(image_bytes) -> Receipt. Stub until models land (spec §4, Lane I)."""
from receipt.schema import Item, Receipt

EXTRACTOR = "stub"


def extract(image_bytes: bytes) -> Receipt:
    # ponytail: fixed fixture so the web lane runs before any model exists; Lane I swaps in
    # preprocess -> ocr -> rows -> extractor -> assemble -> categorize.
    return Receipt(
        merchant="GREEN VALLEY MARKET", address="1420 J STREET, SACRAMENTO, CA 95814",
        date="2026-11-08", time="17:42", subtotal="20.54", tax="0.78", total="21.32",
        category="groceries",
        items=[
            Item(name="ORG BANANAS 2LB", qty="1", price="1.98", category="groceries"),
            Item(name="WHOLE MILK 1GAL", qty="1", price="3.49", category="groceries"),
            Item(name="SOURDOUGH BREAD", qty="1", price="4.29", category="groceries"),
            Item(name="PAPER TOWELS 6PK", qty="1", price="7.99", category="household"),
            Item(name="DISH SOAP", qty="1", price="2.79", category="household"),
        ],
    )
