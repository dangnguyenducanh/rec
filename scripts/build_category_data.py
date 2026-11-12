"""Build data/categories/{items,merchants}.csv from already-labelled public data (spec §4 stage 6).

No hand labels anywhere: Instacart departments, CORD menu names (every CORD receipt is a restaurant),
Souslab US restaurant menu items (drink sections left out), name-suggestion-index brand tags and
OpenStreetMap POI tags. Columns: name,category,source,split.
Run: .venv/bin/python -s scripts/build_category_data.py
"""
import collections
import csv
import hashlib
import json
import random
import re
import time
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

OUT = Path("data/categories")
RAW = OUT / "raw"
UA = {"User-Agent": "receipt-reader-research/0.1"}
NSI_URL = "https://cdn.jsdelivr.net/npm/name-suggestion-index@8.0.20261101/dist/json/nsi.min.json"
SOUSLAB_URL = ("https://huggingface.co/datasets/AnyStackLabsdev/souslab-us-restaurant-menus/resolve/"
               "95c59c394211d1ef839f5941df95d8bcd93fae1b/menu_items_sample.csv")
# Menu sections holding drinks: their items would contradict Instacart's drinks labels.
DRINK_SECTION = re.compile(r"drink|beverage|\bbev|coffee|\btea|wine|beer|liquor|cocktail|spirit|soda|juice|smoothie"
                           r"|shake|\bbar\b|bottle|happy hour|lemonade|espresso|latte", re.I)
OVERPASS = "https://overpass-api.de/api/interpreter"
CITIES = {"seattle": "Q5083", "chicago": "Q1297", "houston": "Q16555"}  # wikidata ids of the city areas

DEPT = {d: "groceries" for d in ["produce", "dairy eggs", "bakery", "meat seafood", "deli", "frozen", "pantry",
                                 "canned goods", "dry goods pasta", "breakfast", "snacks", "international", "bulk"]}
DEPT |= {"beverages": "drinks", "alcohol": "drinks", "household": "household", "pets": "household",
         "babies": "household", "personal care": "health"}  # "missing" / "other": leftover bins, mixed goods
TAG = {("shop", v): "groceries" for v in ["supermarket", "convenience", "greengrocer"]}
TAG |= {("amenity", v): "dining" for v in ["restaurant", "fast_food", "cafe"]}
TAG |= {("amenity", "bar"): "drinks", ("amenity", "pub"): "drinks", ("shop", "alcohol"): "drinks"}
TAG |= {("shop", v): "household" for v in ["hardware", "doityourself", "houseware"]}
TAG |= {("amenity", "pharmacy"): "health", ("shop", "chemist"): "health", ("amenity", "fuel"): "transport_fuel"}
TAG |= {("shop", v): "clothing" for v in ["clothes", "shoes"]}
TAG |= {("shop", v): "electronics" for v in ["electronics", "mobile_phone"]}
TAG |= {("shop", v): "services" for v in ["hairdresser", "laundry", "dry_cleaning"]}
# Lookup-only tags (source suffix "-lk"): exact-name lookup at serving time, never classifier training or
# receipt-style variants. "general" (department/variety/wholesale stores) is not a classifier class either.
LOOKUP = {("shop", v): "general" for v in ["department_store", "variety_store", "general", "wholesale"]}
LOOKUP |= {("shop", v): "other" for v in ["books", "stationery", "gift", "toys"]}
LOOKUP |= {("amenity", "car_wash"): "transport_fuel", ("shop", "car_repair"): "transport_fuel", ("shop", "tyres"): "transport_fuel"}
LOOKUP |= {("shop", "optician"): "health", ("shop", "cosmetics"): "health", ("shop", "computer"): "electronics",
           ("shop", "furniture"): "household", ("shop", "beauty"): "services"}
ALL_TAGS = TAG | LOOKUP
NSI_REGIONS = {"us", "001", "ca", "gb", "ie", "au", "nz", "my", "sg", "in", "ph"}  # English-language receipts
# Tie-break when a name has two categories with the same count.
PRIO = ["general", "groceries", "dining", "drinks", "clothing", "electronics", "household", "services", "other",
        "health", "transport_fuel"]
SIZES = ["1LB", "2LB", "3LB", "8OZ", "12OZ", "16OZ", "32OZ", "64OZ", "1GAL", "HGAL", "12CT", "6PK", "500G", "1KG", "2L"]


def majority(c: collections.Counter, prio=PRIO) -> str:
    """Most frequent category; ties go to the earlier `prio` entry, else to the first seen."""
    return max(c, key=lambda k: (c[k], -prio.index(k) if k in prio else 0))


def norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def split_of(key: str) -> str:
    b = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "train" if b < 70 else "val" if b < 85 else "test"


def receipt_style(name: str, size: bool = True) -> str:
    """'Organic Bananas' -> e.g. 'ORGNC BANA 2LB': uppercase, vowels dropped or words cut to 3-5
    letters, a size appended (items), truncated to 16-20 chars. Deterministic per name."""
    rng = random.Random(zlib.crc32(name.encode()))
    words = []
    for w in re.findall(r"[A-Z0-9]+", name.upper()):
        if len(w) > 4 and w.isalpha():
            x = rng.random()
            w = w[0] + re.sub("[AEIOU]", "", w[1:]) if x < 0.4 else w[:rng.randint(3, 5)] if x < 0.8 else w
        words.append(w)
    s, limit = " ".join(words), rng.randint(16, 20)
    if size and rng.random() < 0.5:
        sz = rng.choice(SIZES)
        return s[:limit - len(sz) - 1].strip() + " " + sz
    return s[:limit].strip()


def get(url: str, data: bytes | None = None, timeout: int = 120) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=UA), timeout=timeout) as f:
        return f.read()


def instacart_rows():
    from huggingface_hub import hf_hub_download
    p = {f: hf_hub_download("attik/Instacart-Market-Basket-Analysis", f, repo_type="dataset",
                            local_dir=RAW / "instacart") for f in ["products.csv", "departments.csv"]}
    dept = {r["department_id"]: r["department"] for r in csv.DictReader(open(p["departments.csv"]))}
    for r in csv.DictReader(open(p["products.csv"])):
        if cat := DEPT.get(dept[r["department_id"]]):
            yield r["product_name"], cat, "instacart", None


def cord_rows():
    for split in ["train", "val"]:  # never test: CORD test is the dining sanity check
        for line in open(f"data/gold/cord/{split}.jsonl"):
            for it in (json.loads(line)["gold"] or {}).get("items", []):
                yield it["name"], "dining", "cord", None


def menu_rows():
    """US restaurant menu items (Souslab sample) -> dining, so English dishes are not only CORD's Indonesian ones.
    One vote per name: a dish on many menus ("Cheese", "Eggs") must not outvote its Instacart label in finish()."""
    p = RAW / "souslab" / "menu_items_sample.csv"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(get(SOUSLAB_URL))
    seen = set()
    for r in csv.DictReader(open(p, newline="")):
        name = re.sub(r"^\s*[A-Z]?\d+[A-Z]?[.#)]\s*|^[A-Z]{1,2}#\s*", "", r["item_name"]).strip()  # "12. ", "A3) ", "B# "
        if not DRINK_SECTION.search(r["section_name"]) and norm(name) not in seen:
            seen.add(norm(name))
            yield name, "dining", "souslab", None


def nsi_rows():
    p = RAW / "nsi.min.json"
    if not p.exists():
        p.write_bytes(get(NSI_URL))
    nsi = json.loads(p.read_text())["nsi"]
    for (k, v), cat in ALL_TAGS.items():
        for it in nsi.get(f"brands/{k}/{v}", {}).get("items", []):
            inc = {re.split(r"[-.]", str(x))[0] for x in it["locationSet"].get("include", [])}  # us-tx.geojson -> us
            if inc & NSI_REGIONS:
                name = it["tags"].get("name") or it["displayName"]
                key = norm(it["tags"].get("brand") or name)
                # The classifier keeps US/world brand names only (other regions lowered its val); the rest is lookup-only.
                yield name, cat, "nsi" if (k, v) in TAG and inc & {"us", "001"} else "nsi-lk", key
                # matchNames only for shops: aliases on fuel/pharmacy entries ("costco wholesale") mislabel the shop
                for n in it.get("matchNames", []) if k == "shop" else []:
                    yield n, cat, "nsi-lk", key


def osm_rows():
    shops = "|".join(v for k, v in ALL_TAGS if k == "shop")
    amen = "|".join(v for k, v in ALL_TAGS if k == "amenity")
    for city, qid in CITIES.items():
        p = RAW / f"overpass_{city}.json"
        if not p.exists():
            q = (f'[out:json][timeout:180];area["wikidata"="{qid}"]->.a;'
                 f'(nwr(area.a)["shop"~"^({shops})$"]["name"];nwr(area.a)["amenity"~"^({amen})$"]["name"];);out tags;')
            try:
                p.write_bytes(get(OVERPASS, urllib.parse.urlencode({"data": q}).encode(), timeout=200))
                time.sleep(5)
            except Exception as e:  # Overpass is often at capacity: skip the city quietly
                print(f"overpass {city}: skipped ({e})")
                continue
        for e in json.loads(p.read_text())["elements"]:
            t = e.get("tags", {})
            kv = next((kv for kv in [("amenity", t.get("amenity")), ("shop", t.get("shop"))] if kv in ALL_TAGS), None)
            if kv and t.get("name"):
                yield t["name"], ALL_TAGS[kv], "osm-lk" if kv in LOOKUP else "osm", norm(t.get("brand") or t["name"])


def finish(rows, size: bool, prio=PRIO) -> list[list[str]]:
    """One row per normalized name (majority category, one count per source row), split by group key
    (chain brand or name), then receipt-style variants of the train names only (not of lookup-only rows).
    The source is that of the winning category, a classifier source ahead of a lookup-only one."""
    by = {}
    for name, cat, src, key in rows:
        n = norm(name)
        if n:
            e = by.setdefault(n, [name, collections.Counter(), {}, key or n])
            e[1][cat] += 1
            e[2][cat] = min(e[2].get(cat, src), src, key=lambda s: s.endswith("-lk"))
    out = [[name, c := majority(cats, prio), srcs[c], split_of(key)] for name, cats, srcs, key in by.values()]
    held = {norm(r[0]) for r in out if r[3] != "train"}
    for name, cat, src, sp in list(out):
        v = receipt_style(name, size)
        if sp == "train" and not src.endswith("-lk") and norm(v) != norm(name) and norm(v) not in held:
            out.append([v, cat, src + "-rs", "train"])
    return out


def write(path: Path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "category", "source", "split"])
        w.writerows(rows)
    c = collections.Counter((r[3], r[1]) for r in rows)
    print(path, len(rows), {s: {k: n for (sp, k), n in sorted(c.items()) if sp == s} for s in ["train", "val", "test"]})


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    write(OUT / "items.csv", finish([*instacart_rows(), *cord_rows(), *menu_rows()], size=True, prio=()))  # items: first seen wins
    write(OUT / "merchants.csv", finish([*nsi_rows(), *osm_rows()], size=False))
