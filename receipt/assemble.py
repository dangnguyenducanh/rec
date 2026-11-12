"""Stage 5: token labels + rows -> Receipt JSON (shared by every extractor, interfaces §2)."""
import datetime as dt
import re
import string
from itertools import groupby
from decimal import ROUND_HALF_UP, Decimal

from receipt import rows
from receipt.schema import Item, Receipt

_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
_MON = (r"(JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUNE?|JULY?|AUG(?:UST)?"
        r"|SEPT?(?:EMBER)?|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)\.?(?![A-Z])")
_D = r"(?<!\d)(\d{1,2})(?:ST|ND|RD|TH)?"
# 19xx/20xx may run into a glued time ("4/4/20127:33AM"); otherwise 2 digits, maybe "'16"
_Y = r"['´’]?((?:19|20)\d{2}|\d{2}(?!\d))"
_DATE_RES = [  # tried in turn; kind says which group is year/month/day ("num": order from dayfirst)
    (re.compile(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)"), "ymd"),
    (re.compile(_D + r"[-/., ]*" + _MON + r"[-/., ]*" + _Y), "dmy"),
    # "JUL19'14", "AUG262017"; not "JUL2023" / "DECEMBER2016" (no day)
    (re.compile(_MON + r"[-/., ]*" + _D + r"(?:[-/., ]+|(?=['´’]|(?:19|20)\d\d))" + _Y), "mdy"),
    (re.compile(r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.]" + _Y), "num"),
]


def norm_money(s: str | None) -> str | None:
    """'$1,234.5' -> '1234.50'; '1.234,50' -> '1234.50'; '60.000' -> '60000.00'; '5.00-' -> '-5.00'."""
    if not s:
        return None
    nums = list(re.finditer(r"\d[\d.,]*", s))
    if not nums:
        return None
    m = nums[-1]  # ponytail: last number in the span wins ("2 x 3.50" -> 3.50); "1 234.50" reads as 234.50
    tok = m.group().rstrip(".,")
    neg = s[:m.start()].rstrip().endswith("-") or (
        s[m.end():].lstrip().startswith("-") and not m.group().endswith((",", ".")))  # "60.000,-" is not negative
    seps = [c for c in tok if c in ".,"]
    if seps:
        dec = seps[-1]
        whole, _, frac = tok.rpartition(dec)
        if len(set(seps)) == 1 and len(frac) == 3:  # only one kind and a final 3-digit group: thousands
            whole, frac = tok, ""
        tok = re.sub(r"[.,]", "", whole) + ("." + frac if frac else "")
    if len(tok) > 15:  # OCR junk (phone/card numbers); also keeps quantize under Decimal's 28 digits
        return None
    v = Decimal(tok).quantize(Decimal("0.01"), ROUND_HALF_UP)
    return str(-v if neg else v)


def norm_date(s: str | None, dayfirst: bool = False) -> str | None:
    """Most receipt date formats -> 'YYYY-MM-DD'; two-digit years are 20xx; invalid -> None."""
    if not s:
        return None
    t = s.upper()
    for rx, kind in _DATE_RES:
        for m in rx.finditer(t):
            a, b, c = m.groups()
            if kind == "ymd":
                y, mo, d = a, b, c
            elif kind == "dmy":
                y, mo, d = c, _MONTHS[b[:3]], a
            elif kind == "mdy":
                y, mo, d = c, _MONTHS[a[:3]], b
            else:
                y, mo, d = c, (b if dayfirst else a), (a if dayfirst else b)
            year = int(y) + (2000 if len(y) == 2 else 0)
            tries = [(int(mo), int(d))] + ([(int(d), int(mo))] if kind == "num" else [])
            for mo_, d_ in tries:  # numeric: swap day/month when the hinted order is impossible
                try:
                    return dt.date(year, mo_, d_).isoformat()
                except ValueError:
                    pass
    return None


def norm_time(s: str | None) -> str | None:
    """'5:42 PM' -> '17:42'; '17:42:10' -> '17:42'; invalid -> None."""
    if not s:
        return None
    for sep in ":.":  # prefer h:mm so "16.02.2023 14:02" picks 14:02
        m = re.search(rf"(?<!\d)(\d{{1,2}})\s*{re.escape(sep)}\s*(\d{{2}})(?:\s*[:.]\s*\d{{2}})?(?:\s*([AP])\.?\s*M\b)?",
                      s.upper())
        if m:
            h, mi, ap = int(m[1]), int(m[2]), m[3]
            if ap:
                if not 1 <= h <= 12:
                    return None
                h = h % 12 + (12 if ap == "P" else 0)
            return f"{h:02d}:{mi:02d}" if h < 24 and mi < 60 else None
    return None


def norm_text(s: str | None) -> str | None:
    """Uppercase, all whitespace removed, edge punctuation stripped; empty -> None."""
    if s is None:
        return None
    return "".join(s.upper().split()).strip(string.punctuation) or None


def _qty(s: str) -> str | None:
    """'2 x' -> '2', '1.00' -> '1'; weights keep their decimals ('0,896' -> '0.896'), as CORD gold does."""
    m = re.search(r"\d+(?:[.,]\d+)?", s)
    if not m:
        return None
    v = m.group().replace(",", ".")
    return str(int(float(v))) if float(v).is_integer() else v


# a count written into the name: "(1x) TOBIKO", "Chicken Burrito (2 @6.70)"
# (?!\d) keeps sizes like "4X4-8FT"; ponytail: a name that really starts with "2XL" loses "2X" to qty
_NAME_QTY = [re.compile(r"^\(?\s*(\d{1,3})\s*[xX×](?!\d)\s*\)?\s*(?=\S)"),
             re.compile(r"\s*\(?\s*(\d{1,3})\s*@\s*\$?\s*[\d.,]+\s*\)?\s*$")]


def build(words: list[dict], labels: list[str], dayfirst: bool = False) -> Receipt:
    quads = [w["quad"] for w in words]
    row_ids = rows.group(quads)
    seq = [(row_ids[i], labels[i] or "O", words[i]["text"]) for i in rows.order(quads, row_ids)]
    by = {}  # label -> texts of its contiguous runs, in reading order
    for lab, g in groupby(seq, key=lambda x: x[1]):
        by.setdefault(lab, []).append(" ".join(t for *_, t in g))

    def first(lab, f, *a):
        return next((v for t in by.get(lab, []) if (v := f(t, *a)) is not None), None)

    def last(lab):
        return next((v for t in reversed(by.get(lab, [])) if (v := norm_money(t)) is not None), None)

    def total():
        # last TOTAL run, skipping ones that repeat a TAX amount and preferring 2-decimal ones;
        # each filter is dropped when it would leave nothing
        c = [(t, v) for t in by.get("TOTAL", []) if (v := norm_money(t)) is not None]
        taxes = {norm_money(t) for t in by.get("TAX", [])}
        c = [x for x in c if x[1] not in taxes] or c
        c = [x for x in c if re.search(r"\d[.,]\d\d(?!\d)", x[0])] or c
        return c[-1][1] if c else None

    date = first("DATE", norm_date, dayfirst)
    # first address block only: more than one unlabelled row before the next ADDRESS word starts another (bill-to)
    addr = [(r, t) for r, lab, t in seq if lab == "ADDRESS"]
    n = next((i for i in range(1, len(addr)) if addr[i][0] > addr[i - 1][0] + 2), len(addr))
    return Receipt(
        merchant=by.get("MERCHANT", [None])[0],
        address=" ".join(t for _, t in addr[:n]) or None,
        date=dt.date.fromisoformat(date) if date else None,
        time=first("TIME", norm_time),
        subtotal=last("SUBTOTAL"), tax=last("TAX"), tips=last("TIPS"), total=total(),
        items=_items(seq),
    )


def _items(seq: list[tuple[int, str, str]]) -> list[Item]:
    """One item per row with ITEM_NAME, qty/price from the same row. Name-only rows wait for the next
    adjacent qty/price row (a name wrapped above its price). Exception: a name-only row right under a
    one-line priced item continues that item (a name wrapped below its price), unless the row after it
    is a name-less qty/price row. A name-less qty/price row with no name waiting fills the price of a
    priceless item right above it, else pairs with the next row (its qty, or its name and price).
    Items still without a price are dropped (modifier / sub-lines)."""
    per_row = {}  # row -> ([names], [qtys], [prices]), runs within the row in x order
    slot = {"ITEM_NAME": 0, "ITEM_QTY": 1, "ITEM_PRICE": 2}
    for (r, lab), g in groupby(seq, key=lambda x: x[:2]):
        if lab in slot:
            per_row.setdefault(r, ([], [], []))[slot[lab]].append(" ".join(t for *_, t in g))
    items, pending, pend_row, last_row, one_line = [], [], None, None, False
    orphan = None  # (row, qty, prices) of a name-less qty/price row with no name waiting above it
    for r in sorted(per_row):
        names, qtys, prices = per_row[r]
        prices = [p for t in prices if (p := norm_money(t)) is not None]
        qty = next((q for t in qtys if (q := _qty(t))), None)
        if pending and r != pend_row + 1:  # gap: the waiting name stands alone
            items.append(Item(name=" ".join(pending)))
            pending = []
        orph, orphan = (orphan if orphan and orphan[0] == r - 1 else None), None
        if (prices or qty is not None) and not names and not pending:
            if items and last_row == r - 1 and items[-1].price is None and prices:  # price wrapped below its item
                items[-1].price = prices[-1]
                items[-1].qty = items[-1].qty or qty
            else:
                orphan = (r, qty, prices)
            continue
        if prices or qty is not None:
            if pending or names:
                if orph and qty is None:  # qty printed on the row above ("3X" over "NAME 19,500")
                    qty = orph[1]
                name = " ".join(pending + names)
                for rx in _NAME_QTY:
                    if (m := rx.search(name)) and len(m.group()) < len(name):
                        qty, name = qty or m[1], name[:m.start()] + name[m.end():]
                items.append(Item(name=name, qty=qty, price=prices[-1] if prices else None,
                                  unit_price=prices[0] if len(prices) > 1 else None))
                last_row, one_line = r, not pending and bool(names)
            pending = []
        elif names and orph and orph[2]:  # name printed under its qty/price row
            items.append(Item(name=" ".join(names), qty=orph[1], price=orph[2][-1]))
            last_row, one_line = r, True
        elif names:
            nxt = per_row.get(r + 1)
            if not pending and one_line and last_row == r - 1 and not (nxt and not nxt[0] and (nxt[1] or nxt[2])):
                items[-1].name += " " + " ".join(names)  # ponytail: an item whose price OCR missed lands here too
                last_row = r
            else:
                pending, pend_row = pending + names, r
    if pending:
        items.append(Item(name=" ".join(pending)))
    return [i for i in items if i.price is not None]  # a line without a price is a modifier / sub-line
