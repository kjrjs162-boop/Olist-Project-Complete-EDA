# -*- coding: utf-8 -*-
"""Olist AI — Streamlit assistant for grounded CSV analysis.

The model writes a read-only DuckDB query; the application executes it against
loaded data and returns actual results. Never treat model-generated numbers as
facts unless they came back from the executed query.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

st.set_page_config(page_title="Olist AI | محلل البيانات", page_icon="📊", layout="wide")

DATA_PATH = Path(os.getenv("DATA_PATH", str(BASE_DIR / "Olist_Master_Dataset.csv"))).expanduser()
DB_PATH = Path(os.getenv("CHATBOT_DB_PATH", str(BASE_DIR / "olist_chat.db"))).expanduser()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'Cairo',sans-serif}
.stApp {background:linear-gradient(140deg,#0b1020,#101a2e 55%,#11152a);color:#f8fafc}
.block-container {max-width:1250px;padding-top:1.2rem}
[data-testid="stSidebar"] {background:#0c1223}
[data-testid="stChatMessage"] {background:#152039;border:1px solid #2b3853;border-radius:16px}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="بقرأ ملف البيانات وبجهّز التحليل...")
def load_data(path: str, modified: float) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ملف البيانات غير موجود: {p}\nضع ملف CSV في المجلد أو عيّن DATA_PATH في .env")
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        try:
            df = pd.read_csv(p, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(p, low_memory=False, encoding="latin-1")
    if df.empty:
        raise ValueError("الملف لا يحتوي على صفوف بيانات.")
    df.columns = [str(c).strip() for c in df.columns]
    return df

try:
    DATA = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0)
    LOAD_ERROR = None
except Exception as exc:
    DATA, LOAD_ERROR = None, str(exc)


def db_connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY KEY,title TEXT,updated_at TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT,conversation_id TEXT,role TEXT,content TEXT,FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE)")
    return conn


def new_conversation() -> str:
    cid = str(uuid.uuid4())
    with db_connect() as c:
        c.execute("INSERT INTO conversations VALUES(?,?,?)", (cid, "محادثة جديدة", datetime.now(timezone.utc).isoformat()))
    return cid


def conversations():
    with db_connect() as c:
        return c.execute("SELECT id,title FROM conversations ORDER BY updated_at DESC").fetchall()


def messages(cid: str):
    with db_connect() as c:
        return c.execute("SELECT role,content FROM messages WHERE conversation_id=? ORDER BY id", (cid,)).fetchall()


def save_message(cid: str, role: str, content: str):
    with db_connect() as c:
        c.execute("INSERT INTO messages(conversation_id,role,content) VALUES(?,?,?)", (cid, role, content))
        if role == "user":
            title = content.replace("\n", " ")[:48]
            c.execute("UPDATE conversations SET title=?,updated_at=? WHERE id=? AND title='محادثة جديدة'", (title, datetime.now(timezone.utc).isoformat(), cid))
        else:
            c.execute("UPDATE conversations SET updated_at=? WHERE id=?", (datetime.now(timezone.utc).isoformat(), cid))


def schema_description(df: pd.DataFrame) -> str:
    parts = [f"Dataset name: {DATA_PATH.name}; rows={len(df)}; columns={len(df.columns)}.", "Columns (name — dtype — sample values):"]
    for col in df.columns:
        samples = df[col].dropna().astype(str).head(3).tolist()
        parts.append(f"- {col} — {df[col].dtype} — {samples}")
    return "\n".join(parts)


def ask_model(question: str, df: pd.DataFrame) -> dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("مفتاح OPENAI_API_KEY غير مضبوط. أضفه في ملف .env بجانب chatbot.py.")
    system = f"""You are a careful bilingual assistant and data analyst. Respond in the user's language.
Dataset metadata follows; do not invent columns or claim calculations you have not performed.
{schema_description(df)}
Return ONLY valid JSON with keys: sql, chart_type, x, y, title, explanation.
If the question is about the uploaded dataset, generate one DuckDB SELECT query over table data. For a general-knowledge question unrelated to the dataset, leave sql empty and answer directly and cautiously in explanation. You have no live web access; for time-sensitive facts, say that the answer may need verification. Never claim to have searched the internet.
Never use INSERT/UPDATE/DELETE/DDL, PRAGMA, COPY, ATTACH, external file functions, or multiple statements. Use quoted identifiers when needed. For aggregations use clear aliases. Prefer COUNT(DISTINCT order_id) when the user asks for number of orders and that column exists; otherwise clearly explain the chosen measure. For totals on denormalized joined data, avoid summing repeated order-level values unless the query deduplicates first; explain assumptions. chart_type must be one of none, bar, line, area, scatter, histogram. Set x/y to result column names when applicable. Keep explanation brief and honest. Do not include SQL markdown fences."""
    r = requests.post(f"{OPENAI_BASE_URL}/chat/completions", headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}, json={"model": MODEL, "temperature": 0, "response_format": {"type": "json_object"}, "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}]}, timeout=90)
    if not r.ok:
        detail = r.text[:600]
        raise RuntimeError(f"تعذر الوصول إلى نموذج الذكاء الاصطناعي (HTTP {r.status_code}): {detail}")
    try:
        return json.loads(r.json()["choices"][0]["message"]["content"])
    except Exception as exc:
        raise RuntimeError("رد النموذج لم يكن بصيغة قابلة للقراءة.") from exc


def run_readonly_query(sql: str, df: pd.DataFrame) -> pd.DataFrame:
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        return pd.DataFrame()
    if ";" in sql or not re.match(r"(?is)^\s*(SELECT|WITH)\b", sql):
        raise ValueError("تم رفض استعلام غير آمن؛ المسموح فقط استعلام قراءة واحد.")
    if not re.search(r'(?i)\b(FROM|JOIN)\s+(?:"data"|data)\b', sql):
        raise ValueError("الاستعلام لازم يقرأ من جدول البيانات data.")
    blocked = r"(?i)\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|COPY|ATTACH|DETACH|INSTALL|LOAD|PRAGMA|EXPORT|IMPORT|CALL|EXECUTE|SECRET|READ_CSV|READ_PARQUET|HTTPFS)\b"
    if re.search(blocked, sql):
        raise ValueError("الاستعلام يحتوي أمرًا غير مسموح به وتم إيقافه.")
    con = duckdb.connect(database=":memory:")
    try:
        con.register("data", df)
        return con.execute(sql).df()
    finally:
        con.close()


def automatic_dashboard(df: pd.DataFrame):
    nrows, ncols = df.shape
    order_cols = [c for c in df.columns if c.lower() in ("order_id", "orderid")]
    customer_cols = [c for c in df.columns if c.lower() in ("customer_unique_id", "customer_id", "customerid")]
    a,b,c,d = st.columns(4)
    a.metric("عدد الصفوف", f"{nrows:,}")
    b.metric("عدد الأعمدة", f"{ncols:,}")
    if order_cols:
        value = int(df[order_cols[0]].nunique(dropna=True))
        c.metric("الطلبات الفريدة", f"{value:,}")
    else:
        c.metric("الأعمدة الرقمية", f"{len(df.select_dtypes(include='number').columns):,}")
    if customer_cols:
        d.metric("العملاء الفريدون", f"{df[customer_cols[0]].nunique(dropna=True):,}")
    else:
        d.metric("قيم مفقودة", f"{int(df.isna().sum().sum()):,}")
    with st.expander("استكشف الأعمدة وجودة البيانات", expanded=False):
        profile = pd.DataFrame({"العمود": df.columns, "النوع": [str(x) for x in df.dtypes], "قيم غير فارغة": [int(df[x].notna().sum()) for x in df.columns], "قيم فريدة": [int(df[x].nunique(dropna=True)) for x in df.columns], "مفقود %": [round(float(df[x].isna().mean()*100),2) for x in df.columns]})
        st.dataframe(profile, use_container_width=True, hide_index=True)
        st.caption("ملاحظة: الصفوف قد تمثل عناصر/دفعات متعددة للطلب نفسه؛ لذلك عدد الصفوف ليس بالضرورة عدد الطلبات.")


def render_result(plan: dict[str, Any], result: pd.DataFrame):
    explanation = str(plan.get("explanation", "تم حساب النتيجة مباشرة من البيانات."))
    if explanation:
        st.markdown(explanation)
    if not result.empty:
        st.dataframe(result, use_container_width=True, hide_index=True)
        chart_type = str(plan.get("chart_type", "none")).lower()
        x, y = plan.get("x"), plan.get("y")
        if chart_type != "none" and x in result.columns:
            st.markdown("#### الرسم البياني")
            chart_df = result.set_index(x)
            if chart_type == "bar": st.bar_chart(chart_df[y] if y in result.columns else chart_df)
            elif chart_type == "line": st.line_chart(chart_df[y] if y in result.columns else chart_df)
            elif chart_type == "area": st.area_chart(chart_df[y] if y in result.columns else chart_df)
            elif chart_type == "scatter" and y in result.columns: st.scatter_chart(result, x=x, y=y)
            elif chart_type == "histogram" and y in result.columns: st.bar_chart(result[y].value_counts().sort_index())
        with st.expander("تفاصيل طريقة الحساب"):
            st.caption("النتيجة أدناه ناتجة عن استعلام قرائي منفّذ على الملف، وليست رقمًا متوقعًا من النموذج.")
            st.code(str(plan.get("sql", "")), language="sql")
    else:
        st.info(explanation or "لم أحتج إلى حساب من الجدول لهذا السؤال.")


st.title("📊 Olist AI — مساعد تحليل البيانات")
st.caption("اسأل بالعربي أو بالإنجليزي، اطلب أرقامًا أو رسومًا أو لوحة مؤشرات؛ النتائج الحسابية تُستخرج من ملفك مباشرة.")

if LOAD_ERROR:
    st.error(LOAD_ERROR)
    st.stop()

with st.sidebar:
    st.header("لوحة البيانات")
    st.success(f"تم تحميل {DATA_PATH.name}")
    st.caption(f"{len(DATA):,} صف × {len(DATA.columns):,} عمود")
    st.divider()
    if st.button("＋ محادثة جديدة", use_container_width=True):
        st.session_state.cid = new_conversation()
        st.rerun()
    for conv in conversations():
        if st.button(f"💬 {conv['title']}", key=f"conv_{conv['id']}", use_container_width=True):
            st.session_state.cid = conv["id"]
            st.rerun()
    st.divider()
    st.caption(f"النموذج: {MODEL}")

if "cid" not in st.session_state:
    old = conversations()
    st.session_state.cid = old[0]["id"] if old else new_conversation()

tab_chat, tab_dash = st.tabs(["💬 اسأل بياناتك", "📈 لوحة البيانات"])
with tab_dash:
    automatic_dashboard(DATA)
    numeric = DATA.select_dtypes(include="number")
    if not numeric.empty:
        st.subheader("نظرة سريعة على الأعمدة الرقمية")
        st.dataframe(numeric.describe().T.round(2), use_container_width=True)
with tab_chat:
    for msg in messages(st.session_state.cid):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    prompt = st.chat_input("مثال: اعمل رسم لأكثر 10 فئات حسب عدد الطلبات")
    if prompt:
        save_message(st.session_state.cid, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("بحلل البيانات وبحسب النتيجة..."):
                try:
                    plan = ask_model(prompt, DATA)
                    result = run_readonly_query(str(plan.get("sql", "")), DATA)
                    render_result(plan, result)
                    response_text = str(plan.get("explanation", "تم تنفيذ التحليل وعرض النتيجة."))
                    if not result.empty:
                        response_text += "\n\nالنتيجة والجدول/الرسم ظاهرين فوق."
                except Exception as exc:
                    response_text = f"ماقدرتش أكمل التحليل: {exc}"
                    st.error(response_text)
        save_message(st.session_state.cid, "assistant", response_text)
        st.rerun()

st.caption("لأفضل دقة: اذكر المقياس المطلوب (عدد الصفوف/الطلبات الفريدة/العملاء)، والفترة، وطريقة التجميع. راجع الاستعلام عند القرارات المهمة.")
