import { useEffect, useState, type DragEvent } from "react";
import { getStats, imageUrl, listReceipts, saveReceipt, uploadReceipt } from "./api";
import { CATEGORIES, type Category, type Item, type Receipt, type Stats, type Stored } from "./types";

type View = "scan" | "receipts" | "spending";
const VIEWS: { id: View; label: string }[] = [
  { id: "scan", label: "Scan" },
  { id: "receipts", label: "Receipts" },
  { id: "spending", label: "Spending" },
];

const label = (c: string) => c.replace("_", " and ").replace(/^./, (s) => s.toUpperCase());
const money = (v: string | number | null) => (v == null ? "—" : `$${Number(v).toFixed(2)}`);
// "3.5" -> "3.50", "" -> null; anything else goes to the API as typed and its error is shown.
const toMoney = (v: string | null) => {
  if (v == null || v.trim() === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(2) : v;
};

export default function App() {
  const [view, setView] = useState<View>("scan");
  const [open, setOpen] = useState<Stored | null>(null);
  const [menu, setMenu] = useState(false);

  const go = (v: View) => {
    setOpen(null);
    setView(v);
    setMenu(false);
    window.scrollTo(0, 0);
  };
  const openReceipt = (s: Stored) => {
    setOpen(s);
    window.scrollTo(0, 0);
  };

  return (
    <>
      <header className={`nav${menu ? " is-open" : ""}`}>
        <div className="wrap nav__bar">
          <button className="nav__logo logo-btn" onClick={() => go("scan")}>
            <span className="mark" data-mark="ring" aria-hidden="true" />
            Receipt Reader
          </button>
          <ul className="nav__links" onClick={() => setMenu(false)}>
            {VIEWS.map((v) => (
              <li key={v.id} className="nav__item">
                <button className="nav__link" aria-current={view === v.id && !open ? "page" : undefined} onClick={() => go(v.id)}>
                  {v.label}
                </button>
              </li>
            ))}
          </ul>
          <button className="nav__toggle" aria-label={menu ? "Close menu" : "Open menu"} aria-expanded={menu} onClick={() => setMenu(!menu)}>
            <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
              <path d={menu ? "M4 4l10 10M14 4L4 14" : "M2 5h14M2 9h14M2 13h14"} stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
            </svg>
          </button>
        </div>
      </header>
      <main className="app">
        <div className="wrap">
          {open ? (
            <Editor key={open.id} stored={open} onBack={() => go("receipts")} />
          ) : view === "scan" ? (
            <Scan onDone={openReceipt} />
          ) : view === "receipts" ? (
            <ReceiptList onOpen={openReceipt} onScan={() => go("scan")} />
          ) : (
            <Spending />
          )}
        </div>
      </main>
    </>
  );
}

function Scan({ onDone }: { onDone: (s: Stored) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pick = (f: File | undefined) => {
    if (!f) return;
    setFile(f);
    setError(null);
    setPreview(URL.createObjectURL(f));
  };
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    pick(e.dataTransfer.files[0]);
  };
  const read = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      onDone(await uploadReceipt(file));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="scan">
      <div className="scan__intro">
        <h1 className="t-h2">Scan a receipt</h1>
        <p className="t-lead measure">
          Upload a photo of a paper receipt. You get the store, date, totals and every item, sorted into 10 spending categories.
        </p>
        <ol className="steps">
          <li><span className="step-num">1</span><div><strong>Upload a photo</strong><p className="muted">JPG, PNG or WebP, up to 10 MB.</p></div></li>
          <li><span className="step-num">2</span><div><strong>Check the result</strong><p className="muted">Totals that don’t add up are flagged for review.</p></div></li>
          <li><span className="step-num">3</span><div><strong>Track spending</strong><p className="muted">By category and by month, from every saved receipt.</p></div></li>
        </ol>
      </div>
      <div className="card scan__card m-nolift">
        <label className={`drop${preview ? " has-file" : ""}`} onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
          <input type="file" accept="image/jpeg,image/png,image/webp" onChange={(e) => pick(e.target.files?.[0])} />
          {preview ? (
            <img src={preview} alt="Selected receipt" />
          ) : (
            <span className="drop__empty">
              <svg width="40" height="40" viewBox="0 0 40 40" aria-hidden="true">
                <path d="M11 4h18v32l-3-2-3 2-3-2-3 2-3-2-3 2z" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
                <path d="M15 12h10M15 17h10M15 22h6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
              </svg>
              <strong>Choose a receipt photo</strong>
              <span className="muted">or drag it here</span>
            </span>
          )}
        </label>
        <div className="scan__foot">
          <span className="muted t-small file-name">{file ? file.name : "No photo chosen"}</span>
          <button className="btn btn--primary" disabled={!file || busy} onClick={read}>
            {busy ? "Reading receipt…" : "Read receipt"}
          </button>
        </div>
        {error && <p className="error" role="alert">{error}</p>}
      </div>
    </div>
  );
}

function Editor({ stored, onBack }: { stored: Stored; onBack: () => void }) {
  const [saved, setSaved] = useState(stored);
  const [draft, setDraft] = useState<Receipt>(stored.receipt);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dirty = JSON.stringify(draft) !== JSON.stringify(saved.receipt);
  const r = saved.receipt;

  const set = (patch: Partial<Receipt>) => setDraft((d) => ({ ...d, ...patch }));
  const setItem = (i: number, patch: Partial<Item>) =>
    setDraft((d) => ({ ...d, items: d.items.map((it, j) => (j === i ? { ...it, ...patch } : it)) }));

  const save = async () => {
    setBusy(true);
    setError(null);
    const clean: Receipt = {
      ...draft,
      subtotal: toMoney(draft.subtotal),
      tax: toMoney(draft.tax),
      tips: toMoney(draft.tips),
      total: toMoney(draft.total),
      date: draft.date || null,
      time: draft.time || null,
      items: draft.items.map((it) => ({ ...it, price: toMoney(it.price), unit_price: toMoney(it.unit_price) })),
    };
    try {
      const next = await saveReceipt(saved.id, clean);
      setSaved(next);
      setDraft(next.receipt);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const field = (key: "merchant" | "address" | "subtotal" | "tax" | "tips" | "total", name: string, num = false) => (
    <label className={`field${num ? " field--num" : ""}`}>
      {name}
      <input inputMode={num ? "decimal" : undefined} value={draft[key] ?? ""} onChange={(e) => set({ [key]: e.target.value })} />
    </label>
  );

  return (
    <div className="editor">
      <button className="back" onClick={onBack}>
        <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M11 6H3M6.5 2.5 3 6l3.5 3.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
        All receipts
      </button>
      <div className="editor__head">
        <div>
          <h1 className="t-h2">{r.merchant ?? "Unknown store"}</h1>
          <p className="muted">{[r.date, r.time, money(r.total)].filter(Boolean).join(" · ")}</p>
        </div>
        <div className="pills">
          <span className={`pill pill--${r.needs_review ? "warn" : "ok"}`}>{r.needs_review ? "Needs review" : "Checks pass"}</span>
          <span className="pill" title="Which extractor read this receipt">Read by {saved.extractor} extractor</span>
        </div>
      </div>

      <div className="editor__grid">
        <figure className="card photo m-nolift">
          <img src={imageUrl(saved.id)} alt="Uploaded receipt" />
        </figure>

        <div className="card sheet m-nolift">
          {r.needs_review ? (
            <div className="notice notice--warn" role="status">
              <strong>Needs review</strong>
              <ul>{r.review_reasons.map((m) => <li key={m}>{m.replace(/^./, (s) => s.toUpperCase())}.</li>)}</ul>
            </div>
          ) : (
            <div className="notice notice--ok" role="status">
              <strong>All checks pass</strong>
              <span>Items, subtotal, tax and total add up.</span>
            </div>
          )}

          <section className="sheet__part">
            <h2 className="t-card">Store</h2>
            <div className="fields">
              {field("merchant", "Merchant")}
              <label className="field">
                Receipt category
                <select value={draft.category} onChange={(e) => set({ category: e.target.value as Category })}>
                  {CATEGORIES.map((c) => <option key={c} value={c}>{label(c)}</option>)}
                </select>
              </label>
              <div className="field--full">{field("address", "Address")}</div>
              <label className="field">
                Date
                <input type="date" value={draft.date ?? ""} onChange={(e) => set({ date: e.target.value })} />
              </label>
              <label className="field">
                Time
                <input type="time" value={draft.time ?? ""} onChange={(e) => set({ time: e.target.value })} />
              </label>
            </div>
          </section>

          <section className="sheet__part">
            <h2 className="t-card">Items <span className="muted">{draft.items.length}</span></h2>
            <div className="items-scroll">
              <table className="items">
                <thead>
                  <tr><th>Item</th><th className="num">Qty</th><th className="num">Price</th><th>Category</th><th><span className="sr-only">Remove</span></th></tr>
                </thead>
                <tbody>
                  {draft.items.map((it, i) => (
                    <tr key={i}>
                      <td><input aria-label={`Item ${i + 1} name`} value={it.name} onChange={(e) => setItem(i, { name: e.target.value })} /></td>
                      <td><input aria-label={`Item ${i + 1} quantity`} className="num qty" value={it.qty ?? ""} onChange={(e) => setItem(i, { qty: e.target.value || null })} /></td>
                      <td><input aria-label={`Item ${i + 1} price`} className="num price" inputMode="decimal" value={it.price ?? ""} onChange={(e) => setItem(i, { price: e.target.value })} /></td>
                      <td>
                        <select aria-label={`Item ${i + 1} category`} value={it.category} onChange={(e) => setItem(i, { category: e.target.value as Category })}>
                          {CATEGORIES.map((c) => <option key={c} value={c}>{label(c)}</option>)}
                        </select>
                      </td>
                      <td>
                        <button className="icon-btn" aria-label={`Remove item ${i + 1}`} onClick={() => set({ items: draft.items.filter((_, j) => j !== i) })}>
                          <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M2.5 2.5l7 7M9.5 2.5l-7 7" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /></svg>
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <button className="add-item" onClick={() => set({ items: [...draft.items, { name: "", qty: "1", unit_price: null, price: null, category: "other" }] })}>
              + Add item
            </button>
          </section>

          <section className="sheet__part">
            <h2 className="t-card">Totals</h2>
            <div className="fields fields--four">
              {field("subtotal", "Subtotal", true)}
              {field("tax", "Tax", true)}
              {field("tips", "Tips", true)}
              {field("total", "Total", true)}
            </div>
          </section>

          {error && <p className="error" role="alert">{error}</p>}
          <div className="sheet__actions">
            <span className="muted t-small">{dirty ? "Unsaved changes" : "All changes saved"}</span>
            <button className="btn btn--primary" disabled={!dirty || busy} onClick={save}>
              {busy ? "Saving…" : "Save changes"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function ReceiptList({ onOpen, onScan }: { onOpen: (s: Stored) => void; onScan: () => void }) {
  const [rows, setRows] = useState<Stored[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    listReceipts().then(setRows, (e) => setError(e.message));
  }, []);

  if (error) return <p className="error" role="alert">{error}</p>;
  if (!rows) return <p className="muted">Loading receipts…</p>;
  const spent = rows.reduce((s, r) => s + Number(r.receipt.total ?? 0), 0);
  const review = rows.filter((r) => r.receipt.needs_review).length;
  return (
    <div className="page">
      <div className="page__head">
        <h1 className="t-h2">Receipts</h1>
        <button className="btn btn--primary" onClick={onScan}>Scan a receipt</button>
      </div>
      <dl className="stats">
        <div><dt className="t-label muted">Receipts</dt><dd className="t-stat">{rows.length}</dd></div>
        <div><dt className="t-label muted">Spent</dt><dd className="t-stat">{money(spent)}</dd></div>
        <div><dt className="t-label muted">Need review</dt><dd className="t-stat">{review}</dd></div>
      </dl>
      <div className="card table-card m-nolift">
        {rows.length === 0 ? (
          <p className="muted empty">No receipts yet. Scan one to start tracking spending.</p>
        ) : (
          <table className="list">
            <thead>
              <tr><th>Date</th><th>Merchant</th><th>Category</th><th className="num">Total</th><th>Status</th></tr>
            </thead>
            <tbody>
              {rows.map((s) => (
                <tr key={s.id} tabIndex={0} onClick={() => onOpen(s)} onKeyDown={(e) => e.key === "Enter" && onOpen(s)}>
                  <td className="mono">{s.receipt.date ?? "—"}</td>
                  <td className="strong">{s.receipt.merchant ?? "Unknown store"}</td>
                  <td>{label(s.receipt.category)}</td>
                  <td className="num mono">{money(s.receipt.total)}</td>
                  <td><span className={`pill pill--${s.receipt.needs_review ? "warn" : "ok"}`}>{s.receipt.needs_review ? "Needs review" : "Checks pass"}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function Spending() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    getStats().then(setStats, (e) => setError(e.message));
  }, []);

  if (error) return <p className="error" role="alert">{error}</p>;
  if (!stats) return <p className="muted">Loading spending…</p>;
  const itemsTotal = stats.by_category.reduce((s, c) => s + Number(c.amount), 0);
  const max = Math.max(...stats.by_category.map((c) => Number(c.amount)), 1);
  const receipts = stats.by_month.reduce((s, m) => s + m.receipts, 0);
  const top = stats.by_category[0];
  return (
    <div className="page">
      <div className="page__head">
        <h1 className="t-h2">Spending</h1>
      </div>
      <dl className="stats">
        <div><dt className="t-label muted">Item spending</dt><dd className="t-stat">{money(itemsTotal)}</dd></div>
        <div><dt className="t-label muted">Receipts</dt><dd className="t-stat">{receipts}</dd></div>
        <div><dt className="t-label muted">Largest category</dt><dd className="t-stat">{top ? label(top.category) : "—"}</dd></div>
      </dl>
      <div className="spend-grid">
        <section className="card m-nolift">
          <h2 className="t-card">Spending by category</h2>
          <p className="muted t-small">Sum of item prices across all receipts, before tax.</p>
          {stats.by_category.length === 0 && <p className="muted empty">Nothing yet.</p>}
          <ul className="bars">
            {stats.by_category.map((c) => (
              <li key={c.category}>
                <span>{label(c.category)}</span>
                <span className="bar-track"><span className="bar" style={{ width: `${(Number(c.amount) / max) * 100}%` }} /></span>
                <span className="num mono">{money(c.amount)}</span>
              </li>
            ))}
          </ul>
        </section>
        <section className="card m-nolift">
          <h2 className="t-card">Spending by month</h2>
          <p className="muted t-small">Receipt totals, tax included.</p>
          <table className="list list--plain">
            <thead><tr><th>Month</th><th className="num">Receipts</th><th className="num">Total</th></tr></thead>
            <tbody>
              {stats.by_month.map((m) => (
                <tr key={m.month}><td className="mono">{m.month}</td><td className="num mono">{m.receipts}</td><td className="num mono">{money(m.amount)}</td></tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>
    </div>
  );
}
