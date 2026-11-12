import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
// taste-skill system (palette putty-black-teal, type manrope, radius soft, buttons arrow), then app CSS.
import "./taste/system.css";
import "./taste/motion.css";
import "./app.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
