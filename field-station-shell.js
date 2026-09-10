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
setTheme(saved === "light" ? "light" : "dark");
toggle?.addEventListener("click", () => setTheme(root.dataset.theme === "dark" ? "light" : "dark"));
