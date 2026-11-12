"""Record a human-paced walkthrough of the web app -> docs/demo/receipt-reader-demo.mp4.

Needs the API (:18300) and the Vite dev server (:18373) running on an empty database, plus ffmpeg.
    .venv/bin/python scripts/record_demo.py
"""
import random
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo"
SAMPLE = OUT / "sample-receipt.jpg"
APP = "http://localhost:18373"
SIZE = {"width": 1280, "height": 800}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

RECEIPT_LINES = [
    "      GREEN VALLEY MARKET", "   1420 J STREET, SACRAMENTO", "           CA 95814", "",
    "   11/08/2026         17:42", "-" * 30,
    "ORG BANANAS 2LB      1   1.98", "WHOLE MILK 1GAL      1   3.49", "SOURDOUGH BREAD      1   4.29",
    "PAPER TOWELS 6PK     1   7.99", "DISH SOAP            1   2.79", "-" * 30,
    "SUBTOTAL                20.54", "TAX 7.25%                0.78", "TOTAL                   21.32", "",
    "VISA ****4417           21.32", "", "   THANK YOU FOR SHOPPING!",
]

# Fake cursor + click ripple + caption box: headless video does not show the real pointer.
OVERLAY_JS = """
window.addEventListener('DOMContentLoaded', () => {
  const css = document.createElement('style');
  css.textContent = `
    #demo-cursor{position:fixed;left:0;top:0;width:22px;height:22px;z-index:99999;pointer-events:none;
      transition:transform .05s;background:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='22' height='22'><path d='M2 2 L2 18 L7 13 L10 20 L13 19 L10 12 L17 12 Z' fill='black' stroke='white' stroke-width='1.5'/></svg>") no-repeat}
    .demo-ripple{position:fixed;width:30px;height:30px;margin:-15px 0 0 -15px;border-radius:50%;z-index:99998;
      pointer-events:none;border:3px solid rgba(31,111,235,.8);animation:demo-r .5s ease-out forwards}
    @keyframes demo-r{from{transform:scale(.3);opacity:1}to{transform:scale(1.6);opacity:0}}
    #demo-caption{position:fixed;left:50%;bottom:28px;transform:translateX(-50%);z-index:99999;pointer-events:none;
      background:rgba(20,22,26,.88);color:#fff;font:600 17px system-ui,sans-serif;padding:10px 18px;border-radius:10px;
      max-width:80%;text-align:center;transition:opacity .3s}`;
  document.head.appendChild(css);
  const c = document.createElement('div'); c.id = 'demo-cursor'; document.body.appendChild(c);
  const cap = document.createElement('div'); cap.id = 'demo-caption'; cap.style.opacity = 0; document.body.appendChild(cap);
  document.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
  document.addEventListener('mousedown', e => {
    const r = document.createElement('div'); r.className = 'demo-ripple';
    r.style.left = e.clientX + 'px'; r.style.top = e.clientY + 'px';
    document.body.appendChild(r); setTimeout(() => r.remove(), 600);
  }, true);
});
"""


def make_sample():
    font = ImageFont.truetype(FONT, 22)
    paper = Image.new("RGB", (470, 40 + 34 * len(RECEIPT_LINES)), (250, 248, 242))
    d = ImageDraw.Draw(paper)
    for i, line in enumerate(RECEIPT_LINES):
        d.text((22, 24 + 34 * i), line, font=font, fill=(40, 40, 46))
    paper = paper.rotate(2.5, expand=True, fillcolor=(0, 0, 0, 0), resample=Image.BICUBIC)
    photo = Image.new("RGB", (paper.width + 260, paper.height + 220), (92, 74, 60))  # wooden table
    shadow = Image.new("L", photo.size, 0)
    ImageDraw.Draw(shadow).rectangle((140, 125, 140 + paper.width, 125 + paper.height), fill=110)
    photo.paste((30, 22, 16), mask=shadow.filter(ImageFilter.GaussianBlur(18)))
    mask = paper.convert("L").point(lambda p: 255 if p > 0 else 0)
    photo.paste(paper, (130, 110), mask)
    photo.filter(ImageFilter.GaussianBlur(0.6)).save(SAMPLE, quality=88)


class Human:
    def __init__(self, page: Page):
        self.page = page
        self.x, self.y = 640, 400

    def pause(self, s: float):
        self.page.wait_for_timeout(int(s * 1000))

    def caption(self, text: str):
        self.page.evaluate("t => { const c = document.getElementById('demo-caption'); c.textContent = t; c.style.opacity = t ? 1 : 0; }", text)

    def move_to(self, locator):
        locator.scroll_into_view_if_needed()
        box = locator.bounding_box()
        tx = box["x"] + box["width"] * random.uniform(0.35, 0.65)
        ty = box["y"] + box["height"] * random.uniform(0.4, 0.6)
        self.page.mouse.move(tx, ty, steps=random.randint(18, 30))
        self.x, self.y = tx, ty
        self.pause(random.uniform(0.25, 0.45))

    def click(self, locator):
        self.move_to(locator)
        self.page.mouse.down()
        self.pause(0.08)
        self.page.mouse.up()
        self.pause(0.5)

    def retype(self, locator, text: str):
        self.click(locator)
        self.page.keyboard.press("Control+A")
        self.pause(0.2)
        self.page.keyboard.type(text, delay=random.randint(110, 160))
        self.pause(0.5)

    def shot(self, path: Path):
        hide = "v => ['demo-cursor','demo-caption'].forEach(id => document.getElementById(id).style.visibility = v)"
        self.page.evaluate(hide, "hidden")
        self.page.screenshot(path=str(path))
        self.page.evaluate(hide, "visible")

    def scroll(self, dy: int):
        for _ in range(abs(dy) // 60):
            self.page.mouse.wheel(0, 60 if dy > 0 else -60)
            self.pause(0.04)
        self.pause(0.6)


def main():
    random.seed(7)
    make_sample()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport=SIZE, record_video_dir=str(OUT / "_raw"), record_video_size=SIZE,
                                  color_scheme="light")
        ctx.add_init_script(OVERLAY_JS)
        page = ctx.new_page()
        h = Human(page)
        page.goto(APP)
        page.mouse.move(h.x, h.y)
        h.caption("Receipt Reader: photo in, categorized spending out")
        h.pause(2.5)

        h.caption("1 · Pick a receipt photo")
        with page.expect_file_chooser() as fc:
            h.click(page.locator("label.drop"))
        fc.value.set_files(str(SAMPLE))
        h.pause(1.8)

        h.caption("2 · Read it (demo build uses a stub extractor until the models are trained)")
        h.click(page.get_by_role("button", name="Read receipt"))
        page.locator(".notice--ok").wait_for()
        h.pause(2.5)

        h.caption("3 · Store, date, items and categories come back, with the photo beside them")
        h.move_to(page.get_by_label("Merchant"))
        h.pause(0.8)
        h.scroll(420)
        h.move_to(page.get_by_label("Item 5 price"))
        h.pause(1.0)

        h.caption("4 · Edit a value: a typo in a price…")
        h.retype(page.get_by_label("Item 5 price"), "27.9")
        h.scroll(400)
        h.click(page.get_by_role("button", name="Save changes"))
        page.locator(".notice--warn").wait_for()
        h.scroll(-900)
        h.caption("…is caught: items no longer add up to the subtotal")
        h.move_to(page.locator(".notice--warn strong"))
        h.pause(2.8)

        h.caption("5 · Fix it and save again")
        h.scroll(420)
        h.retype(page.get_by_label("Item 5 price"), "2.79")
        h.scroll(400)
        h.click(page.get_by_role("button", name="Save changes"))
        page.locator(".notice--ok").wait_for()
        h.scroll(-900)
        h.move_to(page.locator(".notice--ok strong"))
        h.pause(2.2)
        h.shot(OUT / "editor.png")

        h.caption("6 · Every receipt is saved in Postgres")
        h.click(page.get_by_role("button", name="Receipts", exact=True))
        page.locator("table.list tbody tr").first.wait_for()
        h.move_to(page.locator("table.list tbody tr").first)
        h.pause(2.0)

        h.caption("7 · Spending by category, built from item-level categories")
        h.click(page.get_by_role("button", name="Spending", exact=True))
        page.locator(".bars li").first.wait_for()
        h.pause(1.2)
        h.move_to(page.locator(".bars li").first)
        h.pause(2.5)
        h.shot(OUT / "spending.png")
        h.caption("")
        h.pause(1.0)

        raw = page.video.path()
        ctx.close()
        browser.close()

    mp4 = OUT / "receipt-reader-demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "26", "-preset", "slow", "-movflags", "+faststart", str(mp4)], check=True)
    Path(raw).unlink()
    (OUT / "_raw").rmdir()
    print(mp4)


if __name__ == "__main__":
    main()
