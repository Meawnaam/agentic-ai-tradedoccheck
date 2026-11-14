"""TradeDocCheck -- Streamlit demo UI.

    streamlit run app.py
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from tradedoccheck import review
from tradedoccheck.agents.reporter import ReporterAgent
from tradedoccheck.config import DEFAULT_MODELS, PROJECT_ROOT, load_dotenv
from tradedoccheck.graph import EDGES, ENGINE, NODES
from tradedoccheck.llm import LLMError, get_llm
from tradedoccheck.models import CaseResult
from tradedoccheck.pipeline import Pipeline

load_dotenv()
SAMPLES = PROJECT_ROOT / "data" / "samples"
OUT = PROJECT_ROOT / "outputs"
SEV_ICON = {"HIGH": "🔴 HIGH", "MEDIUM": "🟠 MEDIUM", "LOW": "🟡 LOW"}
NODE_LABEL = {
    "ingest": "📥 ingest — อ่านไฟล์ PDF (ข้อความ + ตาราง)",
    "classify": "🏷️ classify — ClassifierAgent ระบุประเภทเอกสาร",
    "extract": "🧾 extract — ExtractorAgent ดึงฟิลด์ด้วยกฎ",
    "llm_repair": "🛠️ llm_repair — LLM อ่านฟิลด์ที่กฎอ่านไม่ได้ (self-correction)",
    "validate": "✅ validate — ValidatorAgent ตรวจไขว้ 15 รายการ (โค้ดล้วน)",
    "decide": "⚖️ decide — ตัดสิน AUTO_PASS / REVIEW / HOLD",
    "investigate": "🕵️ investigate — InvestigatorAgent เลือกเครื่องมือเองเพื่อหาสาเหตุ",
    "report": "📄 report — สรุปผลและเขียนรายงาน",
}

st.set_page_config(page_title="TradeDocCheck", page_icon="📦", layout="wide")


# ====================================================================== helpers
def graph_dot(path: list[str]) -> str:
    taken = set(zip(path, path[1:]))
    lines = ['digraph G { rankdir=LR; bgcolor="transparent";',
             'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11];']
    for n in NODES:
        fill = "#c7f0d8" if n in path else "#eeeeee"
        lines.append(f'"{n}" [fillcolor="{fill}" color="{"#1a7f4b" if n in path else "#999"}"];')
    for a, b, label in EDGES:
        style = 'penwidth=2.5 color="#1a7f4b"' if (a, b) in taken else 'color="#bbbbbb"'
        lines.append(f'"{a}" -> "{b}" [label="{label}" fontsize=9 {style}];')
    lines.append("}")
    return "\n".join(lines)


def build_llm(provider: str, key: str, model: str):
    if provider == "none":
        return get_llm("none")
    if provider == "gemini" and key:
        os.environ["GEMINI_API_KEY"] = key
    if provider == "anthropic" and key:
        os.environ["ANTHROPIC_API_KEY"] = key
    return get_llm(provider, api_key=key or None, model=model or None)


def fmt(v) -> str:
    if v is None or v == "":
        return "—"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


def comparison_table(r: CaseResult) -> pd.DataFrame:
    docs = {d.doc_type: d.fields for d in r.documents}
    inv, pl, bl = (docs.get(t, {}) for t in ("invoice", "packing_list", "bill_of_lading"))
    codes = {f.code: f.severity for f in r.findings}

    def status(*cs):
        sev = [codes[c] for c in cs if c in codes]
        if not sev:
            return "✅ ตรงกัน"
        return SEV_ICON["HIGH" if "HIGH" in sev else "MEDIUM" if "MEDIUM" in sev else "LOW"]

    rows = [
        ("Invoice number", inv.get("invoice_no"), pl.get("invoice_ref"), None,
         status("INVREF_MISMATCH")),
        ("Seller / Shipper", inv.get("seller"), None, bl.get("shipper"), status("PARTY_SHIPPER")),
        ("Buyer / Consignee", inv.get("buyer"), None, bl.get("consignee"),
         status("PARTY_CONSIGNEE")),
        ("Total amount", inv.get("total_amount"), None, None,
         status("INV_TOTAL", "INV_ARITHMETIC")),
        ("Packages", None, pl.get("total_packages"), bl.get("packages"),
         status("PKG_MISMATCH", "PL_SUM")),
        ("Gross weight (kg)", None, pl.get("gross_weight_kg"), bl.get("gross_weight_kg"),
         status("GW_MISMATCH")),
        ("Net weight (kg)", None, pl.get("net_weight_kg"), None, status("NW_GT_GW")),
        ("Date / On-board date", inv.get("date"), pl.get("date"), bl.get("on_board_date"),
         status("DATE_ORDER")),
        ("Line items (qty)", len(inv.get("items") or []) or None,
         len(pl.get("items") or []) or None, None, status("QTY_MISMATCH", "ITEM_UNMATCHED")),
    ]
    return pd.DataFrame([[a, fmt(b), fmt(c), fmt(d), e] for a, b, c, d, e in rows],
                        columns=["Field", "Invoice", "Packing List", "B/L", "Status"])


def findings_table(r: CaseResult) -> pd.DataFrame:
    return pd.DataFrame([{"Severity": SEV_ICON[f.severity], "Code": f.code, "Issue": f.message,
                          "Expected": fmt(f.expected), "Actual": fmt(f.actual),
                          "Suggested action": f.suggestion} for f in r.findings])


# ====================================================================== result view
def show_result(r: CaseResult, key: str):
    n = {s: sum(f.severity == s for f in r.findings) for s in ("HIGH", "MEDIUM", "LOW")}
    banner = {"AUTO_PASS": st.success, "REVIEW": st.warning, "HOLD": st.error}[r.decision]
    banner(f"**{r.decision}** — {r.summary}")

    c = st.columns(5)
    c[0].metric("Decision", r.decision)
    c[1].metric("🔴 HIGH", n["HIGH"])
    c[2].metric("🟠 MEDIUM", n["MEDIUM"])
    c[3].metric("🟡 LOW", n["LOW"])
    c[4].metric("Documents read", f"{len(r.documents)}/3")

    st.caption(f"Engine: `{r.engine}` · LLM: `{r.llm}` · เส้นทางที่กราฟเลือก: "
               + " → ".join(r.path))
    st.graphviz_chart(graph_dot(r.path))

    tabs = st.tabs(["⚠️ Findings", "🔀 เทียบข้ามเอกสาร", "🧾 ข้อมูลที่ดึงได้",
                    "🕵️ Investigator agent", "👤 Human review", "🧭 Agent trace", "⬇️ Export"])

    with tabs[0]:
        if r.findings:
            st.dataframe(findings_table(r), hide_index=True)
        else:
            st.info("ไม่พบความผิดปกติ — เอกสารทั้ง 3 ใบสอดคล้องกัน")

    with tabs[1]:
        st.dataframe(comparison_table(r), hide_index=True)

    with tabs[2]:
        for d in r.documents:
            label = f"{d.doc_type} — {d.source} (via {d.extraction_method})"
            with st.expander(label + (f" · อ่านไม่ได้: {d.missing}" if d.missing else "")):
                header = {k: fmt(v) for k, v in d.fields.items() if k != "items"}
                st.dataframe(pd.DataFrame(header.items(), columns=["Field", "Value"]),
                             hide_index=True)
                if d.fields.get("items"):
                    st.dataframe(pd.DataFrame(d.fields["items"]).astype(str), hide_index=True)

    with tabs[3]:
        inv = r.investigation
        if not inv:
            st.info("กราฟไม่ได้เรียก InvestigatorAgent เพราะเคสนี้ผ่านอัตโนมัติ (AUTO_PASS)")
        else:
            st.markdown(f"**Mode:** `{inv['mode']}`" +
                        ("  ·  ⚠️ **ควรแจ้งหัวหน้างาน (escalate)**" if inv.get("escalate") else ""))
            st.markdown(inv.get("summary", ""))
            left, right = st.columns(2)
            with left:
                st.markdown("**สาเหตุที่น่าจะเป็น (root causes)**")
                for x in inv.get("root_causes", []):
                    st.markdown(f"- {x}")
            with right:
                st.markdown("**สิ่งที่ต้องทำต่อ (actions)**")
                for i, x in enumerate(inv.get("actions", []), 1):
                    st.markdown(f"{i}. {x}")
            st.markdown("**ลำดับการเรียกเครื่องมือที่ agent เลือกเอง**")
            k = 0
            for s in inv.get("steps", []):
                if s["type"] == "thought":
                    st.caption(f"💭 {s['text'][:300]}")
                    continue
                k += 1
                with st.expander(f"{k}. `{s['tool']}({json.dumps(s['args'], ensure_ascii=False)[:80]})`"):
                    st.code(s["result"], language="json")
            if inv.get("email_to_shipper"):
                st.markdown("**ร่างอีเมลถึงผู้ส่งสินค้า (แก้ไขได้ก่อนส่ง)**")
                email = st.text_area("email", inv["email_to_shipper"], height=260,
                                     label_visibility="collapsed", key=f"email_{key}")
                st.download_button("⬇️ ดาวน์โหลดอีเมล (.txt)", email,
                                   file_name=f"{r.case_id}_email.txt", key=f"dl_email_{key}")

    with tabs[4]:
        st.markdown("ระบบเสนอผลเท่านั้น — **เจ้าหน้าที่เป็นผู้ตัดสินขั้นสุดท้าย** "
                    "และทุกการตัดสินถูกบันทึกเพื่อตรวจสอบย้อนหลัง")
        with st.form(f"review_{key}"):
            reviewer = st.text_input("ชื่อผู้ตรวจ")
            action = st.radio("การตัดสินของเจ้าหน้าที่", review.ACTIONS, horizontal=True,
                              index=0 if r.decision == "AUTO_PASS" else 2)
            comment = st.text_area("หมายเหตุ")
            if st.form_submit_button("💾 บันทึกการตัดสิน"):
                rec = review.save_review(OUT, r.case_id, r.decision, action, reviewer, comment)
                msg = "บันทึกแล้ว" + (" — ⚠️ เป็นการ override ผลของระบบ" if rec["override"] else "")
                st.success(msg)

    with tabs[5]:
        st.dataframe(pd.DataFrame(r.trace).astype(str), hide_index=True)

    with tabs[6]:
        md = ReporterAgent.to_markdown(r)
        st.download_button("⬇️ รายงาน Markdown", md, file_name=f"{r.case_id}.md",
                           key=f"dl_md_{key}")
        st.download_button("⬇️ ผลลัพธ์ JSON",
                           json.dumps(r.to_dict(), indent=2, ensure_ascii=False, default=str),
                           file_name=f"{r.case_id}.json", key=f"dl_json_{key}")
        with st.expander("ดูตัวอย่างรายงาน"):
            st.markdown(md)


# ====================================================================== sidebar
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    providers = ["gemini", "none", "anthropic", "ollama"]
    env_default = os.getenv("TRADEDOC_LLM") or ("gemini" if os.getenv("GEMINI_API_KEY") else "none")
    provider = st.selectbox(
        "LLM", providers, index=providers.index(env_default) if env_default in providers else 1,
        format_func={"gemini": "Google Gemini", "none": "ไม่ใช้ LLM (rule-based)",
                     "anthropic": "Anthropic Claude", "ollama": "Ollama (local)"}.get)
    key = ""
    if provider in ("gemini", "anthropic"):
        env_key = "GEMINI_API_KEY" if provider == "gemini" else "ANTHROPIC_API_KEY"
        key = st.text_input("API key", type="password", value=os.getenv(env_key, ""),
                            help="อ่านจากไฟล์ .env อัตโนมัติถ้ามี")
        if provider == "gemini":
            st.caption("รับ key ได้ที่ [Google AI Studio](https://aistudio.google.com/)")
    model = st.text_input("Model", value=os.getenv("TRADEDOC_MODEL")
                          or DEFAULT_MODELS.get(provider, ""), disabled=provider == "none")
    lang = st.selectbox("ภาษาของสรุปจาก agent", ["Thai", "English"])
    st.divider()
    st.caption(f"Workflow engine: **{ENGINE}**"
               + ("" if ENGINE == "langgraph" else " (ติดตั้ง langgraph เพื่อใช้ของจริง)"))

try:
    LLM = build_llm(provider, key, model)
    llm_error = None
except LLMError as e:
    LLM, llm_error = get_llm("none"), str(e)

# ====================================================================== main
st.title("📦 TradeDocCheck")
st.markdown("ระบบ **Document Intelligence แบบ multi-agent** ตรวจความสอดคล้องของ "
            "**Commercial Invoice · Packing List · Bill of Lading** ก่อนผ่านพิธีการศุลกากร")
if llm_error:
    st.warning(f"{llm_error} — กำลังทำงานแบบ rule-based แทน")

tab_case, tab_batch, tab_log, tab_about = st.tabs(
    ["🔍 ตรวจหนึ่ง shipment", "📊 ตรวจทั้งชุด (batch)", "📝 ประวัติการตัดสิน", "ℹ️ ระบบทำงานอย่างไร"])

with tab_case:
    src = st.radio("แหล่งเอกสาร", ["ชุดตัวอย่าง", "อัปโหลด PDF เอง"], horizontal=True)
    folder = None
    if src == "ชุดตัวอย่าง":
        cases = sorted(p.name for p in SAMPLES.iterdir() if p.is_dir()) if SAMPLES.exists() else []
        if not cases:
            st.error("ไม่พบชุดตัวอย่าง — รัน `python scripts/generate_samples.py` ก่อน")
        else:
            choice = st.selectbox("เลือก shipment", cases, index=min(1, len(cases) - 1))
            folder = SAMPLES / choice
            st.caption("ไฟล์: " + ", ".join(p.name for p in sorted(folder.glob("*.pdf"))))
    else:
        files = st.file_uploader("อัปโหลด Invoice, Packing List และ B/L (PDF)", type=["pdf"],
                                 accept_multiple_files=True)
        case_name = st.text_input("ชื่อเคส", value="uploaded_case")
        if files:
            tmp = Path(st.session_state.setdefault("tmpdir", tempfile.mkdtemp())) / case_name
            tmp.mkdir(parents=True, exist_ok=True)
            for old in tmp.glob("*.pdf"):
                old.unlink()
            for f in files:
                (tmp / Path(f.name).name).write_bytes(f.getbuffer())
            folder = tmp

    if st.button("🚀 เริ่มตรวจเอกสาร", type="primary", disabled=folder is None):
        pipe = Pipeline(LLM, lang=lang)
        result = None
        with st.status("LangGraph workflow กำลังทำงาน...", expanded=True) as status:
            t0 = time.time()
            for node, state in pipe.stream_case(folder):
                if node == "__result__":
                    result = state
                    break
                st.write(NODE_LABEL.get(node, node))
            status.update(label=f"เสร็จใน {time.time() - t0:.1f} วินาที", state="complete",
                          expanded=False)
        pipe.save(result, OUT)
        st.session_state["result"] = result

    if st.session_state.get("result") is not None:
        st.divider()
        show_result(st.session_state["result"], key="single")

with tab_batch:
    st.markdown("รันทุกเคสในชุดตัวอย่างพร้อมกัน เพื่อดูว่าระบบคัดเคสที่ต้องให้คนดูออกมาได้อย่างไร")
    if st.button("▶️ รันทั้งชุด"):
        pipe = Pipeline(LLM, lang=lang)
        rows, t0 = [], time.time()
        bar = st.progress(0.0)
        case_dirs = sorted(p for p in SAMPLES.iterdir() if p.is_dir())
        for i, cdir in enumerate(case_dirs, 1):
            r = pipe.process_case(cdir)
            pipe.save(r, OUT)
            rows.append({"Case": r.case_id, "Decision": r.decision,
                         "HIGH": sum(f.severity == "HIGH" for f in r.findings),
                         "MEDIUM": sum(f.severity == "MEDIUM" for f in r.findings),
                         "LOW": sum(f.severity == "LOW" for f in r.findings),
                         "Path": " → ".join(r.path)})
            bar.progress(i / len(case_dirs))
        st.session_state["batch"] = (rows, time.time() - t0)
    if st.session_state.get("batch"):
        rows, secs = st.session_state["batch"]
        df = pd.DataFrame(rows)
        auto = int((df["Decision"] == "AUTO_PASS").sum())
        c = st.columns(3)
        c[0].metric("Shipments", len(df))
        c[1].metric("ผ่านอัตโนมัติ", auto)
        c[2].metric("เวลาที่ใช้", f"{secs:.1f} s")
        st.dataframe(df, hide_index=True)
        st.bar_chart(df.set_index("Case")[["HIGH", "MEDIUM", "LOW"]])

with tab_log:
    recs = review.load_reviews(OUT)
    if recs:
        st.dataframe(pd.DataFrame(recs[::-1]), hide_index=True)
    else:
        st.info("ยังไม่มีการบันทึกการตัดสินจากเจ้าหน้าที่")

with tab_about:
    st.graphviz_chart(graph_dot(NODES))
    st.markdown("""
| Agent | หน้าที่ | ใช้ LLM? |
|---|---|---|
| **ClassifierAgent** | ระบุว่า PDF เป็น Invoice / Packing List / B/L | เฉพาะเมื่อกฎจำไม่ได้ |
| **ExtractorAgent** | ดึงฟิลด์และรายการสินค้าเป็นข้อมูลโครงสร้าง | node `llm_repair` เติมฟิลด์ที่กฎอ่านไม่ได้ |
| **ValidatorAgent** | ตรวจไขว้ 15 รายการ (ยอดเงิน ปริมาณ น้ำหนัก ชื่อคู่ค้า วันที่ HS code) | **ไม่ใช้เลย** — ตรวจสอบย้อนหลังได้ |
| **InvestigatorAgent** | agent ที่ LLM **เลือกเครื่องมือเอง** (10 tools) เพื่อหาสาเหตุ แนะนำการแก้ และร่างอีเมล | ใช่ (มี fallback แบบกฎ) |
| **ReporterAgent** | ตัดสิน AUTO_PASS / REVIEW / HOLD และเขียนรายงาน | เฉพาะการเรียบเรียงสรุป |

**หลักการออกแบบ:** LLM ช่วย *เข้าใจ* เอกสารที่ไม่เป็นมาตรฐานและ *อธิบาย* ผล
แต่การตัดสินเรื่องเงิน น้ำหนัก และปริมาณใช้โค้ดที่ให้ผลเหมือนเดิมทุกครั้ง และเจ้าหน้าที่เป็นผู้ตัดสินขั้นสุดท้าย
""")
