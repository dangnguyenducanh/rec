"""Gold-side normalizers in scripts/prepare_data.py (CORD money/qty, SROIE dates)."""
from scripts.prepare_data import iso_date, money, qty


def test_money():
    cases = {"60.000": "60000.00", "75,000": "75000.00", "1,234.5": "1234.50", "62.000,00": "62000.00",
             "Rp. 200.000": "200000.00", "Rp 45.545.45": "45545.45", "18 000": "18000.00",
             "Pb1 10% 7,000": "7000.00", "-6.42": "-6.42", "RM 10.60": "10.60", "$8.20": "8.20",
             "1.00.": "1.00", "0": "0.00", "-": None, "": None, None: None}
    for s, want in cases.items():
        assert money(s) == want, (s, money(s))


def test_qty():
    for s, want in {"2 x": "2", "x1": "1", "1.0": "1", "*4 Pcs": "4", "0,896X": "0.896", None: None}.items():
        assert qty(s) == want, (s, qty(s))


def test_iso_date_dayfirst():
    cases = {"25/12/2018": "2018-12-25", "25-12-2018": "2018-12-25", "25.12.18": "2018-12-25",
             "25 DEC 2018": "2018-12-25", "27/MAR/2018": "2018-03-27", "28-FEB-2018": "2018-02-28",
             "2018-12-25": "2018-12-25", "20180304": "2018-03-04", "25032018": "2018-03-25",
             "OCT 3, 2016": "2016-10-03", "4/22/2018": "2018-04-22", "(06/12/2016)": "2016-12-06",
             "TOTAL": None, "31/02/2018": None}
    for s, want in cases.items():
        assert iso_date(s, dayfirst=True) == want, (s, iso_date(s, dayfirst=True))
    assert iso_date("12/25/2018") == "2018-12-25"
