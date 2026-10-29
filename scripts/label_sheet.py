"""Usage: uv run python scripts/label_sheet.py NAME  [--port 8765].

A local labelling page for one hand-label sheet (config/hand_label_study.yaml).
It serves data/hand_labels/sheet_NAME.csv to a page on 127.0.0.1 and writes
every label back to that file as it is given, in the format the scorer reads.
It reads the labeller's own sheet only: never key.json, so the checker's
verdict and the condition stay hidden.
"""

import csv
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from settings import DATA_DIR, load_api_study

SPEC = load_api_study("hand_label_study")
FIELDS = ["item", "slice", "question", "reference_answer", "model_answer", "label", "note"]
LABELS = ("", "0", "1", "?")
NOTE_MAX = 500


def _args() -> tuple[str, int]:
    args = sys.argv[1:]
    port = 8765
    if "--port" in args:
        i = args.index("--port")
        port = int(args[i + 1])
        del args[i : i + 2]
    if len(args) != 1 or args[0] not in SPEC["labellers"]:
        raise SystemExit(f"usage: label_sheet.py NAME, NAME one of {', '.join(SPEC['labellers'])}")
    return args[0], port


NAME, PORT = _args()
SHEET = Path(os.environ.get("DRIFTWATCH_LABEL_DIR", DATA_DIR / "hand_labels")) / f"sheet_{NAME}.csv"
LOCK = threading.Lock()


def read_sheet() -> list[dict[str, str]]:
    """The sheet's rows, in its own order."""
    with SHEET.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_sheet(rows: list[dict[str, str]]) -> None:
    """Rewrite the sheet atomically, columns as the builder wrote them."""
    tmp = SHEET.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(SHEET)


class Handler(BaseHTTPRequestHandler):
    """GET / the page, GET /sheet the rows, POST /save one label."""

    def _send(self, code: int, body: bytes, kind: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (http.server's name)
        """Serve the page or the sheet."""
        if self.path == "/":
            self._send(200, PAGE.replace("__NAME__", NAME).encode(), "text/html; charset=utf-8")
        elif self.path == "/sheet":
            rows = [{k: r[k] for k in FIELDS} for r in read_sheet()]
            self._send(200, json.dumps(rows).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        """Save one item's label and note."""
        if self.path != "/save":
            self._send(404, b"not found", "text/plain")
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        item, label, note = body.get("item"), body.get("label"), body.get("note", "")
        if label not in LABELS or not isinstance(note, str) or len(note) > NOTE_MAX:
            self._send(400, b"bad label or note", "text/plain")
            return
        with LOCK:
            rows = read_sheet()
            hits = [r for r in rows if r["item"] == item]
            if len(hits) != 1:
                self._send(400, b"unknown item", "text/plain")
                return
            hits[0]["label"], hits[0]["note"] = label, note.replace("\n", " ").strip()
            write_sheet(rows)
        self._send(200, b"ok", "text/plain")

    def log_message(self, *_: object) -> None:
        """Quiet: the page shows what was saved."""


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hand labels: __NAME__</title>
<style>
:root{--bg:#f7f6f2;--panel:#fff;--ink:#14213d;--muted:#4a5568;--line:#dde1e6;--accent:#b4531a;
--ok:#1f6f43;--okbg:#e5f2ea;--bad:#a3261b;--badbg:#f8e6e3;--unk:#5b4a8a;--unkbg:#ece8f6;--code:#f1f2f4}
@media (prefers-color-scheme:dark){:root{--bg:#12161f;--panel:#1b2130;--ink:#e8ebf1;--muted:#a3acbb;
--line:#2c3446;--accent:#e8a262;--ok:#7fd1a0;--okbg:#18392a;--bad:#f0a199;--badbg:#3d1f1c;--unk:#c5b8f0;
--unkbg:#2c2545;--code:#232a3a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 -apple-system,"Segoe UI",Roboto,Arial,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 20px}
.top{display:flex;align-items:center;gap:16px;flex-wrap:wrap;max-width:1180px;margin:0 auto}
h1{font-size:17px;margin:0;font-weight:650}
.bar{flex:1;min-width:160px;height:8px;background:var(--line);border-radius:99px;overflow:hidden}
.bar>div{height:100%;background:var(--accent);width:0;transition:width .2s}
.count{font-variant-numeric:tabular-nums;color:var(--muted);font-size:14px}
.status{font-size:13px;color:var(--muted);min-width:70px;text-align:right}
main{max-width:1180px;margin:0 auto;padding:20px;display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:20px}
@media (max-width:900px){main{grid-template-columns:1fr}}
main>*{min-width:0}
.meta{display:flex;gap:10px;align-items:center;margin-bottom:12px;flex-wrap:wrap}
.pill{font-size:13px;font-weight:600;padding:2px 10px;border-radius:99px;background:var(--code);color:var(--muted)}
.slice{background:var(--accent);color:var(--bg)}
section.box{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-bottom:14px}
section.box h2{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
pre{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;font:14.5px/1.55 ui-monospace,Menlo,Consolas,monospace}
#question pre{max-height:320px;overflow:auto}
#reference{border-left:4px solid var(--accent)}
#reference pre{font-size:17px;font-weight:600}
#answer pre{max-height:420px;overflow:auto}
aside{position:sticky;top:76px;align-self:start}
.btns{display:grid;gap:10px}
button{font:inherit;cursor:pointer;border-radius:10px;border:1px solid var(--line);background:var(--panel);color:var(--ink);padding:12px 14px;text-align:left;display:flex;justify-content:space-between;align-items:center}
button kbd{font:12px ui-monospace,Menlo,monospace;border:1px solid var(--line);border-radius:5px;padding:1px 6px;color:var(--muted)}
button.l1.on{background:var(--okbg);border-color:var(--ok);color:var(--ok);font-weight:650}
button.l0.on{background:var(--badbg);border-color:var(--bad);color:var(--bad);font-weight:650}
button.lq.on{background:var(--unkbg);border-color:var(--unk);color:var(--unk);font-weight:650}
button:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.nav{display:flex;gap:8px;margin-top:14px}
.nav button{flex:1;justify-content:center;padding:9px}
label.note{display:block;margin-top:16px;font-size:13px;color:var(--muted)}
input[type=text]{width:100%;margin-top:6px;font:inherit;padding:9px 10px;border-radius:8px;border:1px solid var(--line);background:var(--panel);color:var(--ink)}
.gold{margin-top:8px;font-size:13px;padding:6px 10px;justify-content:center}
.rules{margin-top:18px;font-size:13.5px;color:var(--muted);background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.rules b{color:var(--ink)}
.rules p{margin:0 0 6px}
.done{display:none;margin:0 0 14px;padding:12px 16px;border-radius:12px;background:var(--okbg);color:var(--ok);font-weight:600}
.err{color:var(--bad)}
</style></head><body>
<header><div class="top">
<h1>Hand labels · __NAME__</h1>
<div class="bar" aria-hidden="true"><div id="fill"></div></div>
<span class="count" id="count">0 / 0</span>
<span class="status" id="status" role="status"></span>
</div></header>
<main>
<div>
<div class="done" id="done">All items labelled. The sheet is saved; you can close this tab.</div>
<div class="meta"><span class="pill slice" id="slice"></span><span class="pill" id="item"></span><span class="pill" id="pos"></span></div>
<section class="box" id="question"><h2>Question</h2><pre id="q"></pre></section>
<section class="box" id="reference"><h2>Reference answer</h2><pre id="r"></pre></section>
<section class="box" id="answer"><h2>Model answer</h2><pre id="a"></pre></section>
</div>
<aside>
<div class="btns">
<button class="l1" data-l="1">Correct <kbd>1</kbd></button>
<button class="l0" data-l="0">Wrong <kbd>0</kbd></button>
<button class="lq" data-l="?">Can't tell <kbd>? or /</kbd></button>
</div>
<div class="nav">
<button id="prev">← Prev</button><button id="next">Next →</button><button id="skip" title="Next unlabelled">Next blank</button>
</div>
<label class="note" for="note">Note (optional, saved on Enter or when you move on)
<input type="text" id="note" maxlength="500" autocomplete="off"></label>
<button class="gold" id="gold">Mark “gold wrong” <kbd>g</kbd></button>
<div class="rules" id="rules"></div>
</aside>
</main>
<script>
const RULES = {
  gsm8k: "Judge the <b>final number</b> the model gives, not its working. “$18” or “18 dollars” is correct for 18.",
  triviaqa: "Judge <b>meaning</b>, not format. “Paris, the Trojan prince” is correct for “Paris”.",
  squad2: "“unanswerable” means the passage does not contain the answer, and the model should <b>say so</b>. Otherwise the answer must match the reference span in meaning.",
  banking77: "The model's label must be the <b>reference intent</b>. Small formatting differences are fine.",
};
const GENERAL = "<p><b>1</b> correct given the reference · <b>0</b> wrong · <b>?</b> can't tell.</p><p>Label on your own; 10 items are also on someone else's sheet.</p><p>Keys: 1, 0, or ? (or /) to label and move on, ← → to browse, n for the note, g for gold wrong.</p>";
let rows = [], i = 0;
const $ = id => document.getElementById(id);

async function load() {
  rows = await (await fetch("/sheet")).json();
  const first = rows.findIndex(r => !r.label);
  i = first < 0 ? 0 : first;
  show();
}
function progress() {
  const n = rows.filter(r => r.label).length;
  $("count").textContent = `${n} / ${rows.length}`;
  $("fill").style.width = `${100 * n / rows.length}%`;
  $("done").style.display = n === rows.length ? "block" : "none";
}
function show() {
  const r = rows[i];
  $("slice").textContent = r.slice;
  $("item").textContent = r.item;
  $("pos").textContent = `${i + 1} of ${rows.length}`;
  $("q").textContent = r.question;
  $("r").textContent = r.reference_answer;
  $("a").textContent = r.model_answer || "(empty answer)";
  $("note").value = r.note || "";
  document.querySelectorAll(".btns button").forEach(b => b.classList.toggle("on", b.dataset.l === r.label));
  $("rules").innerHTML = `<p>${RULES[r.slice] || ""}</p>` + GENERAL;
  $("q").parentElement.scrollTop = 0; $("a").parentElement.scrollTop = 0;
  $("question").querySelector("pre").scrollTop = 0; $("answer").querySelector("pre").scrollTop = 0;
  progress();
}
async function save(r) {
  $("status").textContent = "saving…"; $("status").classList.remove("err");
  try {
    const res = await fetch("/save", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({item: r.item, label: r.label, note: r.note || ""})});
    if (!res.ok) throw new Error(await res.text());
    $("status").textContent = "saved";
  } catch (e) {
    $("status").textContent = "NOT saved"; $("status").classList.add("err");
  }
}
function keepNote() {
  const r = rows[i], v = $("note").value.trim();
  if ((r.note || "") !== v) { r.note = v; save(r); }
}
function go(j) { keepNote(); i = Math.max(0, Math.min(rows.length - 1, j)); show(); }
function label(l) {
  const r = rows[i]; r.label = l; r.note = $("note").value.trim(); save(r);
  const next = rows.findIndex((x, k) => k > i && !x.label);
  if (next >= 0) go(next); else if (i < rows.length - 1) go(i + 1); else show();
}
document.querySelectorAll(".btns button").forEach(b => b.onclick = () => label(b.dataset.l));
$("prev").onclick = () => go(i - 1);
$("next").onclick = () => go(i + 1);
$("skip").onclick = () => { const k = rows.findIndex(r => !r.label); if (k >= 0) go(k); };
$("gold").onclick = () => {
  const v = $("note").value;
  if (!/gold wrong/i.test(v)) $("note").value = (v ? v + "; " : "") + "gold wrong";
  keepNote();
};
$("note").addEventListener("keydown", e => { if (e.key === "Enter") { keepNote(); $("note").blur(); } e.stopPropagation(); });
document.addEventListener("keydown", e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "1" || e.key === "0") { e.preventDefault(); label(e.key); }
  else if (e.key === "?" || e.key === "/" || e.code === "Slash") { e.preventDefault(); label("?"); }
  else if (e.key === "ArrowLeft") go(i - 1);
  else if (e.key === "ArrowRight") go(i + 1);
  else if (e.key === "n") { e.preventDefault(); $("note").focus(); }
  else if (e.key === "g") $("gold").click();
});
window.addEventListener("beforeunload", keepNote);
load();
</script></body></html>
"""


def main() -> None:
    """Serve the page until interrupted."""
    if not SHEET.exists():
        raise SystemExit(f"{SHEET} is missing")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"labelling {SHEET.name} at {url}  (Ctrl-C to stop; every label is saved as you go)")  # noqa: T201
    threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
