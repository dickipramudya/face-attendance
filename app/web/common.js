// Shared header, clock and helpers for every page.
function header(active) {
  const pages = [["/", "Live"], ["/register", "Register"], ["/people", "People"], ["/report", "Report"], ["/settings", "Settings"]];
  const nav = pages.map(([href, t]) => `<a href="${href}" class="${href === active ? "on" : ""}">${t}</a>`).join("");
  document.body.insertAdjacentHTML("afterbegin", `
    <header>
      <div class="brand"><span id="org">Face Attendance</span><small>Offline face recognition · runs on a $30 Android TV box</small></div>
      <nav>${nav}</nav>
      <div class="clock"><b id="clk"></b><span class="muted" id="dt"></span></div>
    </header>`);
  const tick = () => {
    const d = new Date();
    document.getElementById("clk").textContent = d.toLocaleTimeString("en-GB");
    document.getElementById("dt").textContent = d.toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long", year: "numeric" });
  };
  tick(); setInterval(tick, 1000);
  fetch("/api/settings").then(r => r.json()).then(s => document.getElementById("org").textContent = s.org_name);
}

const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function showMsg(el, text, ok) {
  el.textContent = text;
  el.className = "msg " + (ok ? "ok" : "err");
}
