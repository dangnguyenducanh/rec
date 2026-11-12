import type { Receipt, Stats, Stored } from "./types";

export const API_URL = (import.meta.env.VITE_API_URL ?? "http://localhost:18300").replace(/\/$/, "");

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(API_URL + path, init);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = body?.detail;
    throw new Error(typeof detail === "string" ? detail : `Request failed (${res.status})`);
  }
  return res.json();
}

export const uploadReceipt = (file: File) => {
  const form = new FormData();
  form.append("file", file);
  return call<Stored>("/receipts", { method: "POST", body: form });
};
export const listReceipts = () => call<Stored[]>("/receipts");
export const saveReceipt = (id: number, receipt: Receipt) =>
  call<Stored>(`/receipts/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(receipt),
  });
export const getStats = () => call<Stats>("/stats");
export const imageUrl = (id: number) => `${API_URL}/receipts/${id}/image`;
