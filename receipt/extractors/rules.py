"""Stage 4a: regex + spatial anchor rules (spec §4): top-rows merchant + address block, first date/time,
rightmost amount on keyword rows (total/subtotal/tax/tips), priced rows between header and totals = items."""
import re
import statistics

from receipt import rows
from receipt.assemble import norm_date, norm_money

AMT = re.compile(r"(?<![\d:/])-?[$€£￡¥]?(?:\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{2})?|\d*[.,]\d{2})-?(?![\d:/%])")
TIME = re.compile(r"(?<![\d.])\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]\.?M\b)?|(?<![\d.])\d{1,2}[.:]?\d{2}\s?[AP]\.?M\b")
PHONE = re.compile(r"TEL|PHONE|FAX|\d{9,}|\(\d{3}\)|\d{3}[-. ]\d{3}[-. ]\d{4}|\b\d{2,3}-\d{3,4}\s?\d{4}\b")
COMPANY = re.compile(r"SDN|BHD|ENTERPRISE|TRADING|HARDWARE|GIFT|DECO|BOOK|PHARMACY|BAKERY|KEDAI|RESTORAN|HOTEL|\bLTD|\bINC\b|LLC|\bCO\b|RESTAURANT|CAFE|MARKET|STORE|MART\b|SHOP")
JUNK_TOP = re.compile(r"WELCOME|RECEIPT|INVOICE|COPY|THANK|WWW|HTTP|@|CUSTOMER|ORDER|^\W*$")
HEAD_STOP = re.compile(r"TEL|PHONE|FAX|INVOICE|RECEIPT|CASHIER|SERVER|TABLE|WWW|HTTP|@|GST|REG|DATE|STORE ?#|"
                       r"TRANS|ORDER|CHECK|GUEST|HOST|TERMINAL|MANAGER|OPEN|HOURS")
COMPANY_NO = re.compile(r"ROC|CO\.?\s?(?:REG|NO)|\d{4,}-[A-Z]\b|\(\d+-[A-Z]\)")
SUBTOTAL = re.compile(r"SUB\W?\s?TOTAL")
TOTAL = re.compile(r"TOTAL|AMOUNT\s?DUE|BALANCE\s?DUE|AMOUNT\s?PAYABLE|GRAND")
NOT_TOTAL = re.compile(r"SUB|ITEM|QTY|QUANTITY|SAV|DISC|POINT|EXCL|ROUNDING|TENDER|CASH|CHANGE|CARD|VISA|NUMBER")
TAX = re.compile(r"TAX|GST|VAT|HST|PST")
NOT_TAX = re.compile(r"INVOICE|\bID\b|REG|INCL|EXCL|TAXABLE|EXEMPT|SUMMARY|NO\.?\s?:|TOTAL(?!\s?TAX)")
TIPS = re.compile(r"\bTIPS?\b|GRATUITY")
ITEM_HEADER = re.compile(r"DESCRIPTION|\bQTY\b|\bITEM|\bPRICE\b|\bAMOUNT\b")
NOT_ITEM = re.compile(r"CASH|CHANGE|DISC|SAVING|COUPON|VISA|MASTER|CARD|BALANCE|PAYMENT|TENDER|AMEX|DEBIT|CREDIT|"
                      r"POINTS|ROUND|DEPOSIT|AUTH|APPROV")
QTY = re.compile(r"^(?:\d{1,3}|\d+\s?[X@]|[X@]\s?\d+)$")


def _is_amount(t: str) -> bool:
    return bool(AMT.search(t)) and norm_date(t) is None and not TIME.search(t) \
        and sum(c.isalpha() for c in t) <= 3


def _letters(t: str) -> int:
    return sum(c.isalpha() for c in t)


def extract(doc: dict) -> list[str]:
    words = doc["words"]
    labels = ["O"] * len(words)
    if not words:
        return labels
    quads = [w["quad"] for w in words]
    rid = rows.group(quads)
    R = []  # word indices per row, top->bottom, left->right
    for i in rows.order(quads, rid):
        if R and rid[R[-1][-1]] == rid[i]:
            R[-1].append(i)
        else:
            R.append([i])
    up = [w["text"].upper() for w in words]
    text = [" ".join(up[i] for i in r) for r in R]
    amounts = [[i for i in r if _is_amount(up[i])] for r in R]
    height = [statistics.median(abs(words[i]["quad"][7] - words[i]["quad"][1]) for i in r) for r in R]

    def window(pred, skip=()):  # first (row, words) whose smallest contiguous window satisfies pred
        for k, r in enumerate(R):
            for size in (1, 2, 3, 4):
                for s in range(len(r) - size + 1):
                    ws = r[s:s + size]
                    if not any(i in skip for i in ws) and pred(" ".join(up[i] for i in ws)):
                        return k, ws
        return None, []

    # date / time: first match in reading order
    drow, dw = window(lambda t: norm_date(t) is not None)
    for i in dw:
        labels[i] = "DATE"
    trow, tw = window(lambda t: TIME.fullmatch(t.strip(" :")) is not None or (TIME.search(t) and _letters(t) <= 4),
                      skip=set(dw))
    if tw and (p := R[trow].index(tw[-1]) + 1) < len(R[trow]) and re.fullmatch(r"[AP]\.?M\.?", up[R[trow][p]]):
        tw.append(R[trow][p])  # "5:42" "PM" split into two words
    for i in tw:
        labels[i] = "TIME"

    # amounts on keyword rows (value may sit on the next row when the key is alone)
    def amount_for(k):
        if amounts[k]:
            return amounts[k][-1]
        if k + 1 < len(R) and amounts[k + 1] and _letters(text[k + 1]) <= 2:
            return amounts[k + 1][-1]
        return None

    totals, keyrows = [], set()
    for k, t in enumerate(text):
        lab = None
        if SUBTOTAL.search(t):
            lab = "SUBTOTAL"
        elif TAX.search(t) and not NOT_TAX.search(t):
            lab = "TAX"
        elif TIPS.search(t) and "%" not in t and "SUGGEST" not in t:
            lab = "TIPS"
        elif TOTAL.search(t) and not NOT_TOTAL.search(t):
            if (i := amount_for(k)) is not None:
                totals.append(i)
            keyrows.add(k)
            continue
        if lab:
            keyrows.add(k)
            if (i := amount_for(k)) is not None and labels[i] == "O":
                labels[i] = lab
    if totals:  # ponytail: the largest total-row amount wins (grand total >= partial totals)
        best = max(totals, key=lambda i: float(norm_money(up[i]) or 0))
        labels[best] = "TOTAL"

    # merchant: company-looking row in the top 6, else the tallest of the first 3 text rows
    cand = [k for k in range(min(len(R), 8)) if _letters(text[k]) >= 3 and not JUNK_TOP.search(text[k])
            and not PHONE.search(text[k]) and k not in (drow, trow) and not amounts[k]][:6]
    mrow = None
    if cand:
        comp = [k for k in cand if COMPANY.search(text[k])]
        first3 = cand[:3]
        mrow = comp[0] if comp else max(first3, key=lambda k: (height[k] > 1.3 * height[first3[0]], -k))
        while mrow + 1 < len(R) and (text[mrow].endswith("&") or text[mrow].count("(") > text[mrow].count(")")):
            for i in R[mrow]:
                labels[i] = "MERCHANT"
            mrow += 1  # name wrapped onto the next row
        for i in R[mrow]:
            labels[i] = "MERCHANT"

    # address: consecutive rows under the merchant until a phone / header keyword / date / amount row
    last_head = mrow if mrow is not None else -1
    if mrow is not None:
        for k in range(mrow + 1, min(len(R), mrow + 6)):
            t = text[k]
            if COMPANY_NO.search(t) and k == mrow + 1:
                continue
            if PHONE.search(t) or HEAD_STOP.search(t) or k in (drow, trow) or amounts[k] or ":" in t:
                break
            for i in R[k]:
                labels[i] = "ADDRESS"
            last_head = k

    # items: priced rows between the header block and the first totals keyword row
    start = max([last_head] + [k for k in (drow, trow) if k is not None]
                + [k for k in range(len(R)) if ITEM_HEADER.search(text[k]) and not amounts[k]][:1]) + 1
    end = min([k for k in keyrows if k >= start] or [len(R)])
    for k in range(start, end):
        r, t = R[k], text[k]
        if NOT_ITEM.search(t) or labels[r[0]] != "O":
            continue
        if amounts[k] and _letters(t) >= 2:
            labels[amounts[k][-1]] = "ITEM_PRICE"
            for i in r:
                if labels[i] == "O" and QTY.match(up[i]) and i != r[-1]:
                    labels[i] = "ITEM_QTY"
                elif labels[i] == "O" and _letters(up[i]) >= 1:
                    labels[i] = "ITEM_NAME"
        elif not amounts[k] and _letters(t) >= 2 and k + 1 < end and amounts[k + 1] and _letters(text[k + 1]) < 2:
            for i in r:  # name wrapped above an amount-only row
                if _letters(up[i]) >= 1:
                    labels[i] = "ITEM_NAME"
            labels[amounts[k + 1][-1]] = "ITEM_PRICE"
            for i in R[k + 1]:
                if labels[i] == "O" and QTY.match(up[i]):
                    labels[i] = "ITEM_QTY"
    return labels
