"""Download + convert WildReceipt / CORD / SROIE -> data/gold/{dataset}/{split}.jsonl (interfaces §1).

Run from the repo root:  .venv/bin/python -s scripts/prepare_data.py
Also prints counts + label distribution and draws eval/figures/gold_{dataset}.png.
"""
import collections
import datetime as dt
import glob
import io
import json
import re
import sys
import tarfile
import urllib.request
from decimal import Decimal
from pathlib import Path

RAW, GOLD, FIG = Path("data/raw"), Path("data/gold"), Path("eval/figures")
WR_URL = "https://download.openmmlab.com/mmocr/data/wildreceipt.tar"
HEADER = ["merchant", "address", "date", "time", "subtotal", "tax", "tips", "total"]

WR_MAP = {"Store_name_value": "MERCHANT", "Store_addr_value": "ADDRESS", "Date_value": "DATE",
          "Time_value": "TIME", "Prod_item_value": "ITEM_NAME", "Prod_quantity_value": "ITEM_QTY",
          "Prod_price_value": "ITEM_PRICE", "Subtotal_value": "SUBTOTAL", "Tax_value": "TAX",
          "Tips_value": "TIPS", "Total_value": "TOTAL"}  # every other class (keys, Tel, Others, Ignore) -> O
CORD_MAP = {"menu.nm": "ITEM_NAME", "menu.cnt": "ITEM_QTY", "menu.price": "ITEM_PRICE",
            "sub_total.subtotal_price": "SUBTOTAL", "sub_total.tax_price": "TAX",
            "total.total_price": "TOTAL"}
MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


# ---- value normalizers (gold side) ---------------------------------------------------------
# ponytail: local copies of the money/date rules in interfaces §2, because receipt/assemble.py
# is another lane's and not written yet; switch to assemble.norm_* once it lands.

def money(s) -> str | None:
    """'60.000' -> '60000.00', '1,234.5' -> '1234.50', 'Rp 9.500,00' -> '9500.00', '-' -> None.
    A last separator followed by exactly 3 digits is a thousands separator."""
    if s is None:
        return None
    s = str(s)
    nums = re.findall(r"\d[\d.,]*", re.sub(r"(?<=\d) (?=\d{3}\b)", "", s))  # "18 000" -> "18000"
    if not nums:
        return None
    n = nums[-1].rstrip(".,")  # last number: "Pb1 10% 7,000" -> "7,000"
    m = re.fullmatch(r"(.*)[.,](\d+)", n)
    if m and len(m[2]) != 3:
        v = Decimal(re.sub(r"[.,]", "", m[1]) + "." + m[2])
    else:
        v = Decimal(re.sub(r"[.,]", "", n))
    if s.lstrip().startswith("-"):
        v = -v
    return str(v.quantize(Decimal("0.01")))


def qty(s) -> str | None:
    """'2 x' -> '2', 'x1' -> '1', '1.0' -> '1'. Weights keep their decimals ('0,896X' -> '0.896')."""
    m = re.search(r"\d+(?:[.,]\d+)?", str(s or ""))
    if not m:
        return None
    v = m[0].replace(",", ".")
    return str(int(float(v))) if float(v).is_integer() else v


def iso_date(s, dayfirst=False) -> str | None:
    """25/12/2018, 25-12-18, 25.12.18, 25 DEC 2018, 27/MAR/2018, DEC 25, 2018, 2018-12-25, 20181225.
    Numeric dates use the locale order, then the other order when that is impossible (4/22/2018)."""
    s = str(s or "").upper()
    num, alt = ("dmy", "mdy") if dayfirst else ("mdy", "dmy")
    nd = r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}|\d{2})(?!\d)"
    for pat, order in [(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)", "ymd"),
                       (r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)", "ymd"),
                       (r"(?<!\d)(\d{2})(\d{2})(\d{4})(?!\d)", num),  # 25032018
                       (nd, num), (nd, alt),
                       (r"(?<!\d)(\d{1,2})[\s/.-]*([A-Z]{3})[A-Z]*[\s/.,-]*(\d{4}|\d{2})(?!\d)", "dmy"),
                       (r"([A-Z]{3})[A-Z]*[\s.-]*(\d{1,2}),?\s*(\d{4})(?!\d)", "mdy")]:
        m = re.search(pat, s)
        if not m:
            continue
        p = dict(zip(order, m.groups()))
        y = int(p["y"]) + (2000 if len(p["y"]) == 2 else 0)
        mon = MONTHS.get(p["m"][:3]) if p["m"].isalpha() else int(p["m"])
        try:
            return dt.date(y, mon, int(p["d"])).isoformat()
        except (TypeError, ValueError):
            continue
    return None


def valid_gold(d: dict, dropped: collections.Counter) -> dict:
    """Receipt.model_validate; a bad field is set to null (counted in `dropped`), never the record."""
    from pydantic import ValidationError

    from receipt.schema import Receipt
    for _ in range(20):
        try:
            return Receipt.model_validate(d).model_dump(mode="json")
        except ValidationError as e:
            for err in e.errors():
                loc = err["loc"]
                if loc[0] == "items":
                    d["items"][loc[1]][loc[2]] = None
                else:
                    d[loc[0]] = None
                dropped[".".join(str(x) for x in loc if not isinstance(x, int))] += 1
    raise ValueError(f"cannot validate {d}")


def record(dataset, split, key, image, w, h, words, gold, fields, items_labeled, dayfirst=False, roi=None):
    words = [w_ for w_ in words if w_["text"].strip()]  # boxes without text are not words
    return {"id": f"{dataset}/{split}/{key}", "dataset": dataset, "split": split, "image": str(image),
            "width": w, "height": h, "words": words, "roi": roi, "gold": gold, "fields": fields,
            "items_labeled": items_labeled, "dayfirst": dayfirst}


def save_image(b: bytes, path: Path) -> tuple[int, int]:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(b)  # original bytes, no re-encode: boxes stay in this pixel frame
    im = Image.open(io.BytesIO(b))
    assert im.getexif().get(274, 1) == 1, f"{path}: EXIF rotation would move the pixel frame"
    return im.size


def write(dataset, split, recs):
    out = GOLD / dataset / f"{split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---- datasets ------------------------------------------------------------------------------

def wildreceipt():
    root = RAW / "wildreceipt"
    if not (root / "test.txt").exists():
        tar = RAW / "wildreceipt.tar"
        req = urllib.request.Request(WR_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(tar, "wb") as f:
            f.write(r.read())
        with tarfile.open(tar) as t:
            t.extractall(RAW, filter="data")
        tar.unlink()
    names = {}
    for line in open(root / "class_list.txt"):
        idx, name = line.split(maxsplit=1)
        names[int(idx)] = name.strip()
    assert set(WR_MAP) <= set(names.values()), names

    def conv(r, split, i):
        words = []
        for a in r["annotations"]:
            b = a["box"]  # WildReceipt corners run BL, BR, TR, TL -> reverse to TL, TR, BR, BL
            words.append({"text": a["text"], "quad": [b[6], b[7], b[4], b[5], b[2], b[3], b[0], b[1]],
                          "label": WR_MAP.get(names[a["label"]], "O"), "row": None, "group": None})
        return record("wildreceipt", split, f"{i:04d}", root / r["file_name"], r["width"], r["height"],
                      words, None, HEADER, True)

    load = lambda f: sorted((json.loads(l) for l in open(root / f)), key=lambda r: r["file_name"])
    train = load("train.txt")
    splits = {"train": [r for i, r in enumerate(train) if i % 10 != 9],
              "val": [r for i, r in enumerate(train) if i % 10 == 9],  # every 10th sorted file name
              "test": load("test.txt")}
    return {sp: [conv(r, sp, i) for i, r in enumerate(rs)] for sp, rs in splits.items()}


def _hf(repo, local):
    if not glob.glob(f"{local}/data/*.parquet"):
        from huggingface_hub import snapshot_download
        snapshot_download(repo, repo_type="dataset", local_dir=local, allow_patterns=["data/*.parquet"])


def _rows(local, split):
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(f"{local}/data/{split}-*.parquet")):
        yield from pq.read_table(f).to_pylist()


def _quad(q):
    return [q["x1"], q["y1"], q["x2"], q["y2"], q["x3"], q["y3"], q["x4"], q["y4"]]


def cord_gold(p: dict, dropped) -> dict:
    join = lambda v: " ".join(map(str, v)) if isinstance(v, list) else v  # nm/cnt can be lists
    last = lambda v: v[-1] if isinstance(v, list) else v  # money lists: last value, like assemble
    sect = lambda v: {k: x for d in v for k, x in d.items()} if isinstance(v, list) else (v or {})
    menu = p.get("menu") or []
    items = [{"name": join(m["nm"]), "qty": qty(join(m.get("cnt"))), "price": money(last(m.get("price")))}
             for m in ([menu] if isinstance(menu, dict) else menu) if m.get("nm")]
    sub, tot = sect(p.get("sub_total")), sect(p.get("total"))
    return valid_gold({"currency": "IDR", "subtotal": money(last(sub.get("subtotal_price"))),
                       "tax": money(last(sub.get("tax_price"))), "total": money(last(tot.get("total_price"))),
                       "items": items}, dropped)


def cord(dropped):
    local = RAW / "cord" / "hf"
    _hf("naver-clova-ix/cord-v2", local)
    out, keys_as_o, size_diff = {}, 0, 0
    for hf_split, split in [("train", "train"), ("validation", "val"), ("test", "test")]:
        recs = []
        for i, r in enumerate(_rows(local, hf_split)):
            gt = json.loads(r["ground_truth"])
            key = f"{gt['meta'].get('image_id', i):04d}"
            img = RAW / "cord" / split / f"{key}.png"
            w, h = save_image(r["image"]["bytes"], img)
            size_diff += (w, h) != (gt["meta"]["image_size"]["width"], gt["meta"]["image_size"]["height"])
            words = []
            for vl in gt["valid_line"]:
                for wd in vl["words"]:
                    label = CORD_MAP.get(vl["category"], "O")
                    if wd.get("is_key") and label != "O":  # "TOTAL" key word inside a total line
                        label, keys_as_o = "O", keys_as_o + 1
                    words.append({"text": wd["text"], "quad": _quad(wd["quad"]), "label": label,
                                  "row": wd.get("row_id"), "group": vl.get("group_id")})
            roi = _quad(gt["roi"]) if gt.get("roi") else None
            recs.append(record("cord", split, key, img, w, h, words, cord_gold(gt["gt_parse"], dropped),
                               ["subtotal", "tax", "total"], True, roi=roi))
        assert len({r["id"] for r in recs}) == len(recs)
        out[split] = recs
    print(f"cord: {keys_as_o} is_key words inside mapped lines -> O; {size_diff} images differ from meta size")
    return out


def sroie(dropped):
    local = RAW / "sroie" / "hf"
    _hf("jsdnrs/ICDAR2019-SROIE", local)
    out, bad = {}, collections.Counter()
    for split in ["train", "test"]:
        recs = []
        for r in _rows(local, split):
            img = RAW / "sroie" / split / f"{r['key']}.jpg"
            w, h = save_image(r["image"]["bytes"], img)
            words = [{"text": t, "quad": [x1, y1, x2, y1, x2, y2, x1, y2], "label": None, "row": None, "group": None}
                     for t, (x1, y1, x2, y2) in zip(r["words"], r["bboxes"])]
            e = r["entities"]
            date, total = iso_date(e["date"], dayfirst=True), money(e["total"])
            bad["date"] += bool(e["date"]) and date is None
            bad["total"] += bool(e["total"]) and total is None
            gold = valid_gold({"merchant": e["company"] or None, "address": e["address"] or None,
                               "date": date, "total": total, "currency": "MYR"}, dropped)
            recs.append(record("sroie", split, r["key"], img, w, h, words, gold,
                               ["merchant", "address", "date", "total"], False, dayfirst=True))
        assert len({r["id"] for r in recs}) == len(recs), "duplicate SROIE keys"
        out[split] = recs
    print(f"sroie: unparseable gold -> null: {dict(bad)}")
    return out


# ---- check: counts, label distribution, one drawing per dataset -----------------------------

COLORS = {"MERCHANT": "#e6194b", "ADDRESS": "#f58231", "DATE": "#3cb44b", "TIME": "#008080",
          "ITEM_NAME": "#4363d8", "ITEM_QTY": "#911eb4", "ITEM_PRICE": "#f032e6", "SUBTOTAL": "#9a6324",
          "TAX": "#808000", "TIPS": "#46f0f0", "TOTAL": "#000075", "O": "#a9a9a9", None: "#a9a9a9"}


def draw(rec):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(rec["image"]).convert("RGB")
    s = 1400 / im.height
    im = im.resize((round(im.width * s), 1400))
    panel = Image.new("RGB", (520, 1400), "white")
    d, dp = ImageDraw.Draw(im), ImageDraw.Draw(panel)
    font = ImageFont.load_default(size=14)
    for w in rec["words"]:
        pts = [(x * s, y * s) for x, y in zip(w["quad"][0::2], w["quad"][1::2])]
        d.polygon(pts, outline=COLORS[w["label"]], width=1 if w["label"] in ("O", None) else 3)
        if w["label"] not in ("O", None):
            d.text((pts[0][0], pts[0][1] - 14), w["label"], fill=COLORS[w["label"]], font=font)
    y = 10
    dp.text((10, y), rec["id"], fill="black", font=font)
    for lab, c in COLORS.items():
        if lab is not None:
            y += 20
            dp.rectangle((10, y, 30, y + 14), fill=c)
            dp.text((40, y), lab, fill="black", font=font)
    if rec["gold"]:
        g = {k: v for k, v in rec["gold"].items() if v not in (None, [], False, "other")}
        g["items"] = [f"{i['qty']} x {i['name']} = {i['price']}" for i in g.get("items", [])]
        dp.multiline_text((10, y + 30), json.dumps(g, indent=1, ensure_ascii=False)[:2500], fill="black", font=font)
    canvas = Image.new("RGB", (im.width + panel.width, 1400), "white")
    canvas.paste(im)
    canvas.paste(panel, (im.width, 0))
    FIG.mkdir(parents=True, exist_ok=True)
    canvas.save(FIG / f"gold_{rec['dataset']}.png")


def main():
    sys.path.insert(0, ".")  # receipt.schema when run as scripts/prepare_data.py
    dropped = collections.Counter()
    for name, data in [("wildreceipt", wildreceipt()), ("cord", cord(dropped)), ("sroie", sroie(dropped))]:
        labels = collections.Counter()
        for split, recs in data.items():
            write(name, split, recs)
            labels.update(w["label"] for r in recs for w in r["words"])
            n_words = sum(len(r["words"]) for r in recs)
            print(f"{name}/{split}: {len(recs)} receipts, {n_words} words")
        print(f"{name} labels: {dict(labels.most_common())}")
        test = data["test"]
        draw(max(test[:30], key=lambda r: (len({w['label'] for w in r['words']}), len(r["words"]))))
    print(f"gold fields dropped by validation: {dict(dropped)}")


if __name__ == "__main__":
    main()
