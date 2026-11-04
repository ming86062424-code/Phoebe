"""Draw docs/images/architecture.svg, the layered system diagram in docs/architecture.md.

    uv run python -m scripts.draw_architecture

Edit the layout below and rerun; the SVG is written by hand-placed coordinates.
"""
from pathlib import Path
from html import escape

OUT = Path(__file__).resolve().parent.parent / "docs" / "images" / "architecture.svg"
W, H = 1200, 900
parts: list[str] = []


def text_w(s: str, size: float) -> float:
    return sum(size * (1.0 if ord(c) > 0x2E80 else 0.58) for c in s)


def group(x, y, w, h, color, label, dashed=True):
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    parts.append(f'<rect class="g g-{color}" x="{x}" y="{y}" width="{w}" height="{h}" rx="12"{dash}/>')
    parts.append(f'<text class="gl t-{color}" x="{x + 14}" y="{y + 22}">{escape(label)}</text>')


def card(x, y, w, h, color, title, subs=(), icon=None):
    parts.append(f'<rect class="card" x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>')
    tx = x + 14
    if icon:
        s = 30
        parts.append(f'<use href="#{icon}" class="ic-{color}" x="{x + 14}" y="{y + (h - s) / 2}" width="{s}" height="{s}"/>')
        tx = x + 56
    total = 18 + 17 * len(subs)
    ty = y + (h - total) / 2 + 14
    parts.append(f'<text class="ct" x="{tx}" y="{ty}">{escape(title)}</text>')
    for i, s in enumerate(subs):
        parts.append(f'<text class="cs" x="{tx}" y="{ty + 18 + 17 * i}">{escape(s)}</text>')


def edge(points, label=None, lx=None, ly=None, anchor="middle"):
    d = " ".join(f"{px},{py}" for px, py in points)
    parts.append(f'<polyline class="e" points="{d}" marker-end="url(#arrow)"/>')
    if label:
        tw = text_w(label, 12)
        bx = {"middle": lx - tw / 2, "start": lx, "end": lx - tw}[anchor]
        parts.append(f'<rect class="lbg" x="{bx - 4:.1f}" y="{ly - 12}" width="{tw + 8:.1f}" height="17" rx="3"/>')
        parts.append(f'<text class="el" x="{lx}" y="{ly}" text-anchor="{anchor}">{escape(label)}</text>')


def line(points):
    d = " ".join(f"{px},{py}" for px, py in points)
    parts.append(f'<polyline class="e" points="{d}"/>')


# ---- layout ----
# user
card(585, 30, 200, 60, "gray", "使用者", ["瀏覽器"], "i-browser")

# outer box: the machine
parts.append('<rect class="outer" x="30" y="120" width="890" height="740" rx="16" stroke-dasharray="8 5"/>')
parts.append('<text class="ol" x="48" y="146">本機 · Docker Compose</text>')

# row 1
group(60, 170, 390, 150, "orange", "背景工作")
card(80, 205, 350, 100, "orange", "ingest", ["解析文件 → 產生向量 → 寫入 Milvus", "每 2 秒查一次文件佇列", "LiteParse 在這裡執行"], "i-doc")
group(490, 170, 390, 150, "blue", "對外服務 · :8000")
card(510, 205, 350, 100, "blue", "web", ["Gunicorn 2 個 worker + FastAPI", "網頁介面（React）與所有 API", "檢索、組提示詞、串流回答"], "i-server")

# row 2
group(130, 380, 490, 140, "purple", "AI 模型 · 只開放本機")
card(150, 415, 220, 90, "purple", "vllm（選用）", ["LightOnOCR", "GPU"], "i-scan")
card(390, 415, 210, 90, "purple", "embedder", ["embeddinggemma-2", "文字與圖片轉向量"], "i-vector")
group(690, 380, 190, 140, "green", "監控")
card(710, 415, 150, 90, "green", "monitor", ["每 5 秒取樣", "容器、GPU 用量"], "i-chart")

# row 3: data
group(60, 580, 820, 260, "teal", "資料層", dashed=False)
group(80, 612, 370, 213, "teal", "向量資料庫")
card(100, 642, 330, 70, "teal", "milvus", ["向量查詢 + BM25 關鍵字查詢"], "i-db")
card(100, 742, 155, 62, "teal", "etcd", ["設定"])
card(275, 742, 155, 62, "teal", "minio", ["資料檔"])
line([(177, 712), (177, 740)])
line([(352, 712), (352, 740)])
group(470, 612, 390, 213, "teal", "共用儲存 · web 與 ingest 掛同一個 volume")
card(490, 642, 350, 52, "teal", "app.db（SQLite）", ["帳號、對話、知識庫、文件佇列"], "i-db")
card(490, 702, 350, 52, "teal", "monitor.db（SQLite）", ["事件與資源取樣"], "i-db")
card(490, 762, 350, 52, "teal", "volumes/kb", ["原始檔、頁面圖、裁切的圖片"], "i-folder")

# outside docker
group(950, 170, 230, 350, "gray", "Docker 外")
card(970, 205, 190, 62, "gray", "Gemini API", ["雲端聊天模型（預設）"], "i-cloud")
card(970, 280, 190, 62, "gray", "Ollama", ["主機上的聊天模型"], "i-pc")
card(970, 428, 190, 64, "gray", "Docker socket", ["nvidia-smi"], "i-term")

# ---- edges ----
edge([(685, 90), (685, 203)], "HTTP :8000", 685, 162)
edge([(860, 236), (968, 236)], "聊天（串流）", 914, 229)
edge([(860, 290), (968, 310)])
edge([(860, 460), (968, 460)], "讀取用量", 914, 453)
edge([(260, 305), (260, 413)], "整頁圖辨識", 260, 354)
edge([(410, 305), (410, 413)], "chunk 轉向量", 410, 354)
edge([(560, 305), (560, 413)], "問題轉向量", 560, 354)
edge([(95, 305), (95, 578)], "寫入 chunk、更新佇列狀態", 103, 556, "start")
edge([(655, 305), (655, 578)], "查詢、讀寫", 647, 556, "end")
edge([(785, 505), (785, 578)], "寫入取樣", 793, 556, "start")

parts.append(f'<text class="foot" x="30" y="886">箭頭：誰呼叫誰（或讀寫哪裡）。web 和 ingest 不直接互相呼叫，透過 app.db 的文件佇列溝通。</text>')

ICONS = """
<symbol id="i-browser" viewBox="0 0 24 24"><rect x="2.5" y="4" width="19" height="16" rx="2"/><path d="M2.5 9h19"/><path d="M5.5 6.5h.01M8 6.5h.01M10.5 6.5h.01"/></symbol>
<symbol id="i-server" viewBox="0 0 24 24"><rect x="3" y="3.5" width="18" height="7" rx="1.5"/><rect x="3" y="13.5" width="18" height="7" rx="1.5"/><path d="M7 7h.01M7 17h.01M11 7h6M11 17h6"/></symbol>
<symbol id="i-doc" viewBox="0 0 24 24"><path d="M6 2.5h8l5 5v14H6z"/><path d="M14 2.5v5h5"/><path d="M9 12.5h7M9 16.5h7"/></symbol>
<symbol id="i-scan" viewBox="0 0 24 24"><path d="M3 8V3h5M16 3h5v5M21 16v5h-5M8 21H3v-5"/><path d="M7 9.5h10M7 12h10M7 14.5h6"/></symbol>
<symbol id="i-vector" viewBox="0 0 24 24"><circle cx="5" cy="6" r="2.2"/><circle cx="19" cy="6" r="2.2"/><circle cx="12" cy="18.5" r="2.2"/><path d="M7.2 6h9.6M6.1 8l4.8 8.6M17.9 8l-4.8 8.6"/></symbol>
<symbol id="i-chart" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M6.5 15.5l3.5-4 3 3 4.5-6"/></symbol>
<symbol id="i-db" viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></symbol>
<symbol id="i-folder" viewBox="0 0 24 24"><path d="M3 6.5a1.5 1.5 0 0 1 1.5-1.5h4.5l2 2.5h8.5a1.5 1.5 0 0 1 1.5 1.5v9.5a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 18.5z"/></symbol>
<symbol id="i-cloud" viewBox="0 0 24 24"><path d="M7 19h10.5a4.5 4.5 0 0 0 .4-8.98A6.5 6.5 0 0 0 5.6 9.6 4.75 4.75 0 0 0 7 19z"/></symbol>
<symbol id="i-pc" viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="12" rx="1.5"/><path d="M8 20.5h8M12 16v4.5"/></symbol>
<symbol id="i-term" viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 10l3 2.5L7 15M12.5 15H17"/></symbol>
"""

FONT = "'Segoe UI','Microsoft JhengHei','PingFang TC','Noto Sans TC','Noto Sans CJK TC',sans-serif"
COLORS = {  # name: (light stroke, light fill, dark stroke, dark fill)
    "blue": ("#3b6fd4", "#eef3fd", "#6d9eff", "#16223a"),
    "orange": ("#c96a12", "#fdf3e8", "#f0a050", "#33240f"),
    "purple": ("#7b52c7", "#f4effc", "#ad8cf0", "#271d3b"),
    "green": ("#2f8a4a", "#edf7f0", "#5cc47c", "#142b1c"),
    "teal": ("#13808c", "#eaf6f7", "#4cc3cf", "#0f2a2e"),
    "gray": ("#6b7280", "#f5f6f8", "#9aa3af", "#1c2027"),
}


def color_css(dark: bool) -> str:
    out = []
    for name, (ls, lf, ds, df) in COLORS.items():
        s, f = (ds, df) if dark else (ls, lf)
        out.append(f".g-{name}{{stroke:{s};fill:{f}}} .t-{name}{{fill:{s}}} .ic-{name}{{color:{s}}}")
    return "\n".join(out)


STYLE = f"""
text{{font-family:{FONT}}}
.bg{{fill:#ffffff}}
.outer{{fill:none;stroke:#57606a;stroke-width:1.5}}
.ol{{font-size:15px;font-weight:700;fill:#24292f}}
.g{{stroke-width:1.4}}
.gl{{font-size:13.5px;font-weight:700}}
.card{{fill:#ffffff;stroke:#d0d7de;stroke-width:1}}
.ct{{font-size:15px;font-weight:700;fill:#1f2328}}
.cs{{font-size:12.5px;fill:#57606a}}
.e{{fill:none;stroke:#6e7781;stroke-width:1.6}}
.ah{{fill:#6e7781}}
.lbg{{fill:#ffffff;opacity:.92}}
.el{{font-size:12px;fill:#424a53}}
.foot{{font-size:12.5px;fill:#57606a}}
use{{fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}}
{color_css(False)}
@media (prefers-color-scheme: dark) {{
.bg{{fill:#0d1117}}
.outer{{stroke:#8b949e}}
.ol{{fill:#e6edf3}}
.card{{fill:#161b22;stroke:#3d444d}}
.ct{{fill:#e6edf3}}
.cs{{fill:#9198a1}}
.e{{stroke:#8b949e}}
.ah{{fill:#8b949e}}
.lbg{{fill:#0d1117}}
.el{{fill:#c9d1d9}}
.foot{{fill:#9198a1}}
{color_css(True)}
}}
"""

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="title">
<title id="title">Phoebe 系統架構圖</title>
<style>{STYLE}</style>
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path class="ah" d="M0 0L10 5L0 10z"/></marker>
{ICONS}
</defs>
<rect class="bg" width="{W}" height="{H}"/>
{chr(10).join(parts)}
</svg>
"""
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    f.write(svg)
print("wrote", OUT)
