// Mirrors receipt/schema.py (spec §3). Keep the two in sync by hand.
export const CATEGORIES = [
  "groceries", "dining", "drinks", "household", "health",
  "transport_fuel", "clothing", "electronics", "services", "other",
] as const;
export type Category = (typeof CATEGORIES)[number];

export interface Item {
  name: string;
  qty: string | null;
  unit_price: string | null;
  price: string | null;
  category: Category;
}

export interface Receipt {
  merchant: string | null;
  address: string | null;
  date: string | null;
  time: string | null;
  currency: string;
  subtotal: string | null;
  tax: string | null;
  tips: string | null;
  total: string | null;
  category: Category;
  needs_review: boolean;
  review_reasons: string[];
  items: Item[];
}

export interface Stored {
  id: number;
  created_at: string;
  extractor: string;
  receipt: Receipt;
}

export interface Stats {
  by_category: { category: Category; amount: string }[];
  by_month: { month: string; amount: string; receipts: number }[];
}
