"""Normalizers and build() on a tiny synthetic receipt."""
import datetime as dt

from receipt.assemble import build, norm_date, norm_money, norm_text, norm_time


def test_norm_money():
    cases = {"$1,234.50": "1234.50", "1.234,50": "1234.50", "60.000": "60000.00", "Rp 1.500.000": "1500000.00",
             "12.4": "12.40", "RM9.00": "9.00", "1,234.5": "1234.50", "12,40": "12.40", "7": "7.00",
             "5.00-": "-5.00", "-3.00": "-3.00", "60.000,-": "60000.00", "TOTAL 12.40": "12.40",
             "abc": None, "": None, None: None, "9" * 30: None}
    assert {k: norm_money(k) for k in cases} == cases


def test_norm_date():
    us = {"02/16/23": "2023-02-16", "2/16/2023": "2023-02-16", "16 FEB 2023": "2023-02-16",
          "16-Feb-23": "2023-02-16", "16FEB23": "2023-02-16", "2023-02-16": "2023-02-16",
          "2023/2/5": "2023-02-05", "Feb 16, 2023": "2023-02-16", "February 16 2023": "2023-02-16",
          "16th Feb 2023": "2023-02-16", "Date: 02/16/2023 14:02": "2023-02-16",
          "Jul19'14": "2014-07-19", "27Dec´16": "2016-12-27", "DECEMBER2016": None,  # WildReceipt spellings
          "Aug262017": "2017-08-26", "4/4/20127:33AM": "2012-04-04", "JUL2023": None,
          "WEDOCTOBER11,2017": "2017-10-11",
          "13/02/2023": "2023-02-13",  # month 13 impossible -> read day first
          "02/30/2023": None, "MARKET 12": None, "junk": None, None: None}
    assert {k: norm_date(k) for k in us} == us
    assert norm_date("05/04/2023", dayfirst=True) == "2023-04-05"
    assert norm_date("05/04/2023") == "2023-05-04"
    assert norm_date("31.12.22", dayfirst=True) == "2022-12-31"


def test_norm_time_and_text():
    cases = {"5:42 PM": "17:42", "17:42:10": "17:42", "12:05 AM": "00:05", "12:05 PM": "12:05",
             "11:30AM": "11:30", "9.30": "09:30", "16.02.2023 14:02": "14:02", "25:00": None, "x": None}
    assert {k: norm_time(k) for k in cases} == cases
    assert norm_text("  Safe way, Inc. ") == "SAFEWAY,INC"
    assert norm_text(" ... ") is None and norm_text(None) is None


def _w(text, x0, row, w=None):
    x1, y0 = x0 + (w or 12 * len(text)), 40 * row
    return {"text": text, "quad": [x0, y0, x1, y0, x1, y0 + 30, x0, y0 + 30]}


def test_build():
    layout = [  # (text, x, row, label)
        ("SAFE", 0, 0, "MERCHANT"), ("WAY", 60, 0, "MERCHANT"),
        ("12", 0, 1, "ADDRESS"), ("MAIN", 40, 1, "ADDRESS"), ("ST", 100, 1, "ADDRESS"),
        ("SPRINGFIELD", 0, 2, "ADDRESS"),
        ("02/16/23", 0, 3, "DATE"), ("5:42", 200, 3, "TIME"), ("PM", 260, 3, "TIME"),
        ("CHICKEN", 0, 4, "ITEM_NAME"), ("SANDWICH", 100, 4, "ITEM_NAME"), ("WITH", 220, 4, "ITEM_NAME"),
        ("CHEESE", 0, 5, "ITEM_NAME"), ("1", 300, 5, "ITEM_QTY"), ("$", 380, 5, "ITEM_PRICE"),
        ("12.40", 400, 5, "ITEM_PRICE"),  # name wrapped; price on its last line
        ("COKE", 0, 6, "ITEM_NAME"), ("2.00", 400, 6, "ITEM_PRICE"),
        ("SUBTOTAL", 0, 7, "O"), ("14.40", 400, 7, "SUBTOTAL"),
        ("TAX", 0, 8, "O"), ("1.15", 400, 8, "TAX"),
        ("TOTAL", 0, 9, "O"), ("15.55", 400, 9, "TOTAL"),
    ]
    order = [5, 0, 3, 1]  # shuffle input order; build must not depend on it
    words, labels = [], []
    for j in order + [i for i in range(len(layout)) if i not in order]:
        t, x, r, lab = layout[j]
        words.append(_w(t, x, r))
        labels.append(lab)
    rec = build(words, labels)
    assert (rec.merchant, rec.address) == ("SAFE WAY", "12 MAIN ST SPRINGFIELD")
    assert (rec.date, rec.time) == (dt.date(2023, 2, 16), "17:42")
    assert (rec.subtotal, rec.tax, rec.tips, rec.total) == ("14.40", "1.15", None, "15.55")
    assert [(i.name, i.qty, i.price) for i in rec.items] == [
        ("CHICKEN SANDWICH WITH CHEESE", "1", "12.40"), ("COKE", None, "2.00")]


def test_build_item_layouts():
    """Name line then qty/price line; and a wrapped name below a priced line."""
    layout = [("BANANAS", 0, 0, "ITEM_NAME"), ("2", 0, 1, "ITEM_QTY"), ("1.98", 400, 1, "ITEM_PRICE"),
              ("MILK", 0, 2, "ITEM_NAME"), ("3.49", 400, 2, "ITEM_PRICE"), ("2%", 0, 3, "ITEM_NAME")]
    rec = build([_w(t, x, r) for t, x, r, _ in layout], [lab for *_, lab in layout])
    assert [(i.name, i.qty, i.price) for i in rec.items] == [("BANANAS", "2", "1.98"), ("MILK 2%", None, "3.49")]
