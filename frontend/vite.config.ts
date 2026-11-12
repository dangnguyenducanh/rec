import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { port: 18373, strictPort: true },
  preview: { port: 18373, strictPort: true },
});
