const root = document.documentElement;
const key = "field-station-theme";
const toggle = document.getElementById("theme-toggle");

function setTheme(theme) {
  root.dataset.theme = theme;
  try { localStorage.setItem(key, theme); } catch {}
  if (toggle) {
    toggle.textContent = theme === "dark" ? "Light" : "Dark";
    toggle.setAttribute("aria-label", `Use ${theme === "dark" ? "light" : "dark"} theme`);
  }
}

let saved = null;
try { saved = localStorage.getItem(key); } catch {}
// The FIELD STATION public surface is light by default; dark is a deliberate alternate.
setTheme(saved === "dark" ? "dark" : "light");
toggle?.addEventListener("click", () => setTheme(root.dataset.theme === "dark" ? "light" : "dark"));
