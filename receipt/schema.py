"""Contract (spec §3): Receipt/Item models, unified token labels, category list."""
import datetime as dt
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, Field

LABELS = ["MERCHANT", "ADDRESS", "DATE", "TIME", "ITEM_NAME", "ITEM_QTY", "ITEM_PRICE",
          "SUBTOTAL", "TAX", "TIPS", "TOTAL", "O"]

Category = Literal["groceries", "dining", "drinks", "household", "health",
                   "transport_fuel", "clothing", "electronics", "services", "other"]
CATEGORIES = list(get_args(Category))

Money = Annotated[str, Field(pattern=r"^-?\d+\.\d{2}$")]  # 2-decimal string, e.g. "12.40"


class Item(BaseModel):
    name: str
    qty: str | None = None
    unit_price: Money | None = None
    price: Money | None = None
    category: Category = "other"


class Receipt(BaseModel):
    merchant: str | None = None
    address: str | None = None
    date: dt.date | None = None
    time: Annotated[str, Field(pattern=r"^\d{2}:\d{2}$")] | None = None
    currency: str = "USD"
    subtotal: Money | None = None
    tax: Money | None = None
    tips: Money | None = None
    total: Money | None = None
    category: Category = "other"
    needs_review: bool = False
    review_reasons: list[str] = []
    items: list[Item] = []
