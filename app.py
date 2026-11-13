"""
AI Retail Sales Analyzer
Upload sales data -> get summaries, trends, insights, two AI analyses, recommendations,
an interactive dashboard and downloadable reports.

Run:  streamlit run app.py
API key: put GROQ_API_KEY in .streamlit/secrets.toml  OR  set it as an environment variable.
"""
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from groq import Groq
except ImportError:  # app still works without AI if the package is missing
    Groq = None

# Groq retires models often, so the app asks Groq which models are live and picks the best one.
# To force a model, set GROQ_MODEL (secrets or environment variable).
PREFERRED_MODELS = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b",
                    "meta-llama/llama-4-scout-17b-16e-instruct", "qwen/qwen3-32b", "llama-3.1-8b-instant"]
NOT_CHAT = ("whisper", "guard", "tts", "orpheus", "playai", "embed", "compound")
NAVY, DEEP, BLUE, LBLUE = "#14213D", "#1D3557", "#3A86FF", "#2563EB"
PALETTE = [BLUE, "#15803D", "#F59E0B", "#8B5CF6", "#EF4444", "#06B6D4", "#EC4899", "#64748B"]

st.set_page_config(page_title="AI Retail Sales Analyzer", page_icon="🛒", layout="wide")

# ----------------------------------------------------------------------------------
# 1. CUSTOM CSS
# ----------------------------------------------------------------------------------
st.markdown(f"""
<style>
.stApp {{ background:#F0F7FF; color:{NAVY}; }}
[data-testid="stSidebar"] {{ background:{NAVY}; }}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label p,[data-testid="stSidebar"] .stMarkdown p {{ color:#E5EDFF !important; }}
.hero {{ background:{DEEP}; padding:28px 34px; border-radius:18px; margin-bottom:18px;
        box-shadow:0 6px 18px rgba(29,53,87,.25); }}
.hero h1 {{ color:#fff; font-size:2.6rem; margin:0; font-weight:800; }}
.hero p  {{ color:#BBD3FF; font-size:1.15rem; margin:6px 0 0; }}
.kpi {{ border-radius:16px; padding:18px 20px; box-shadow:0 4px 14px rgba(20,33,61,.10); height:100%; }}
.kpi .l {{ font-size:.9rem; font-weight:600; color:{NAVY}; opacity:.8; }}
.kpi .v {{ font-size:1.9rem; font-weight:800; margin:2px 0; }}
.kpi .s {{ font-size:.85rem; color:{NAVY}; opacity:.75; }}
.box {{ border-radius:14px; padding:14px 18px; margin:8px 0; color:{NAVY}; border-left:6px solid; }}
.box b {{ display:block; margin-bottom:2px; }}
.green  {{ background:#D8F3DC; border-color:#15803D; }}
.orange {{ background:#FFE5B4; border-color:#C2410C; }}
.purple {{ background:#EDE9FE; border-color:#7C3AED; }}
.blue   {{ background:#DBEAFE; border-color:{LBLUE}; }}
.card-title {{ font-weight:800; font-size:1.15rem; color:{LBLUE}; margin-bottom:4px; }}
[class*="st-key-card"] {{ background:#fff; border-radius:16px; padding:10px 16px;
        box-shadow:0 4px 14px rgba(20,33,61,.10); }}
.st-key-cardA {{ border-top:6px solid #7C3AED; }}
.st-key-cardB {{ border-top:6px solid {BLUE}; }}
.st-key-cardJudge {{ background:#D8F3DC; border-top:6px solid #15803D; }}
.st-key-cardRec {{ background:#DBEAFE; border-top:6px solid {LBLUE}; }}
.stTabs [data-baseweb="tab"] {{ font-weight:700; }}
.stTabs [aria-selected="true"] {{ color:{LBLUE}; }}
.stButton>button, .stDownloadButton>button {{ background:{BLUE}; color:#fff; border:0;
        border-radius:10px; font-weight:700; padding:.5rem 1.2rem; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ background:{LBLUE}; color:#fff; }}
</style>
""", unsafe_allow_html=True)


def kpi(label, value, sub="", bg="#FFFFFF", color=NAVY):
    return (f'<div class="kpi" style="background:{bg}"><div class="l">{label}</div>'
            f'<div class="v" style="color:{color}">{value}</div><div class="s">{sub}</div></div>')


def box(kind, title, text):
    st.markdown(f'<div class="box {kind}"><b>{title}</b>{text}</div>', unsafe_allow_html=True)


def money(x):
    return f"{x:,.0f}"


# ----------------------------------------------------------------------------------
# 2. DATA LOADING AND CLEANING
# ----------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_file(file_bytes, name):
    """Read a CSV or Excel upload into a DataFrame."""
    import io
    buf = io.BytesIO(file_bytes)
    if name.lower().endswith(".csv"):
        try:
            return pd.read_csv(buf)
        except UnicodeDecodeError:
            buf.seek(0)
            return pd.read_csv(buf, encoding="latin-1")
    return pd.read_excel(buf)


def guess(df, keys, exclude=()):
    """Return the first column whose name contains one of the keywords."""
    for k in keys:
        for c in df.columns:
            if k in c.lower() and c not in exclude:
                return c
    return None


def clean(df, cols):
    """Fix common data errors. Returns the cleaned frame and a list of notes."""
    notes, df = [], df.copy()
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        notes.append(f"Removed {before - len(df)} duplicate rows.")
    for key in ("sales", "profit", "qty"):
        c = cols[key]
        if c:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")
    if cols["date"]:
        df[cols["date"]] = pd.to_datetime(df[cols["date"]], errors="coerce")
        bad = df[cols["date"]].isna().sum()
        if bad:
            notes.append(f"Dropped {bad} rows with unreadable dates.")
            df = df.dropna(subset=[cols["date"]])
    bad = df[cols["sales"]].isna().sum()
    if bad:
        notes.append(f"Dropped {bad} rows with missing or invalid sales values.")
        df = df.dropna(subset=[cols["sales"]])
    for key in ("profit", "qty"):
        c = cols[key]
        if c and df[c].isna().any():
            notes.append(f"Filled {df[c].isna().sum()} missing values in '{c}' with the median.")
            df[c] = df[c].fillna(df[c].median())
    for key in ("product", "category", "region", "customer"):
        c = cols[key]
        if c and df[c].isna().any():
            notes.append(f"Filled {df[c].isna().sum()} missing values in '{c}' with 'Unknown'.")
            df[c] = df[c].fillna("Unknown")
    return df, notes


# ----------------------------------------------------------------------------------
# 3. ANALYSIS HELPERS
# ----------------------------------------------------------------------------------
def group(df, cols, by):
    """Sales (and profit) totals per group, sorted by sales."""
    spec = {"Sales": (cols["sales"], "sum")}
    if cols["profit"]:
        spec["Profit"] = (cols["profit"], "sum")
    g = df.groupby(by).agg(**spec).reset_index().sort_values("Sales", ascending=False)
    if "Profit" in g:
        g["Margin %"] = (g["Profit"] / g["Sales"].replace(0, np.nan) * 100).round(1)
    return g


def time_series(df, cols, rule):
    return df.set_index(cols["date"])[cols["sales"]].resample(rule).sum().reset_index(name="Sales")


def trend_facts(df, cols):
    """Monthly growth, peak/low months and overall direction."""
    if not cols["date"]:
        return None
    m = time_series(df, cols, "MS")
    m["Month"] = m[cols["date"]].dt.strftime("%b %Y")
    m["Growth %"] = m["Sales"].pct_change() * 100
    if len(m) < 2:
        return {"monthly": m, "peak": m.iloc[0], "low": m.iloc[0], "direction": "flat", "slope": 0}
    slope = np.polyfit(range(len(m)), m["Sales"], 1)[0]
    pct = slope / max(m["Sales"].mean(), 1) * 100
    direction = "growing" if pct > 1 else "declining" if pct < -1 else "flat"
    return {"monthly": m, "peak": m.loc[m["Sales"].idxmax()], "low": m.loc[m["Sales"].idxmin()],
            "direction": direction, "slope": pct}


def build_facts(df, cols):
    """Collect real numbers from the data. These are fed to the AI and used to fact-check it."""
    S, P = cols["sales"], cols["profit"]
    f = {"rows": len(df), "total_sales": df[S].sum(), "avg_sales": df[S].mean()}
    f["total_profit"] = df[P].sum() if P else None
    f["margin"] = f["total_profit"] / f["total_sales"] * 100 if P and f["total_sales"] else None
    for key in ("product", "category", "region", "customer"):
        if cols[key]:
            f[key] = group(df, cols, cols[key])
    f["trend"] = trend_facts(df, cols)
    return f


def facts_text(f, cols):
    """Turn the facts into plain text for the AI prompt."""
    t = [f"Rows: {f['rows']}", f"Total sales: {f['total_sales']:.2f}", f"Average sale: {f['avg_sales']:.2f}"]
    if f["total_profit"] is not None:
        t.append(f"Total profit: {f['total_profit']:.2f} (margin {f['margin']:.1f}%)")
    tr = f["trend"]
    if tr:
        t.append(f"Overall trend: {tr['direction']}. Peak month: {tr['peak']['Month']} ({tr['peak']['Sales']:.0f}). "
                 f"Lowest month: {tr['low']['Month']} ({tr['low']['Sales']:.0f}).")
        t.append("Monthly sales:\n" + tr["monthly"][["Month", "Sales"]].tail(24).round(0).to_string(index=False))
    for key in ("product", "category", "region", "customer"):
        if key in f:
            g = f[key]
            t.append(f"Top {key}s:\n{g.head(5).to_string(index=False)}")
            if key == "product":
                t.append(f"Lowest products:\n{g.tail(5).to_string(index=False)}")
    return "\n\n".join(t)


def api_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY") or st.session_state.get("manual_key", "")


def setting(name):
    """Read a setting from Streamlit secrets, then environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


def pick_model(client):
    """Choose a chat model that is currently available on this Groq account."""
    if setting("GROQ_MODEL"):
        return setting("GROQ_MODEL")
    if "model" not in st.session_state:
        available = [m.id for m in client.models.list().data if not any(x in m.id.lower() for x in NOT_CHAT)]
        if not available:
            raise RuntimeError("Your Groq account has no chat models available.")
        st.session_state["model"] = next((m for m in PREFERRED_MODELS if m in available), available[0])
    return st.session_state["model"]


def ask_ai(system, prompt, temperature=0.3):
    """One call to the Groq LLM. Returns text, or raises a readable error."""
    if Groq is None:
        raise RuntimeError("The 'groq' package is not installed. Run: pip install groq")
    key = api_key()
    if not key:
        raise RuntimeError("No Groq API key found. Add GROQ_API_KEY to Streamlit secrets or paste it in the sidebar.")
    client = Groq(api_key=key)
    r = client.chat.completions.create(
        model=pick_model(client), temperature=temperature, max_tokens=3000,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    text = r.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # some models print their reasoning


def fact_check(text, f):
    """Share of key real facts (top product, top category, peak month) that the AI text mentions."""
    checks = []
    if "product" in f:
        checks.append(str(f["product"].iloc[0, 0]).lower())
    if "category" in f:
        checks.append(str(f["category"].iloc[0, 0]).lower())
    if f["trend"]:
        checks.append(f["trend"]["peak"]["Month"].lower())
    if not checks:
        return None
    return sum(c in text.lower() for c in checks) / len(checks)


# ----------------------------------------------------------------------------------
# 4. SIDEBAR: UPLOAD, COLUMN MAPPING, FILTERS
# ----------------------------------------------------------------------------------
st.markdown('<div class="hero"><h1>🛒 AI Retail Sales Analyzer</h1>'
            '<p>Transform Sales Data into Smart Business Insights</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📂 Your Data")
    upload = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if not api_key():
        st.text_input("Groq API key (optional here)", type="password", key="manual_key",
                      help="Better: store GROQ_API_KEY in .streamlit/secrets.toml")

if upload is None:
    box("purple", "👋 Welcome!", "Upload a monthly sales file in the sidebar to begin. "
        "Your file needs at least a <b>sales</b> column. Date, product, category, region, profit and "
        "customer columns unlock more insights.")
    st.stop()

try:
    with st.spinner("Reading your file..."):
        raw = load_file(upload.getvalue(), upload.name)
except Exception as e:
    st.error(f"Could not read this file: {e}. Check that it is a valid CSV or Excel file.")
    st.stop()
if raw.empty:
    st.error("The file has no rows. Upload a file that contains sales records.")
    st.stop()

# Auto-detect columns, but let the user correct them.
with st.sidebar.expander("🧭 Column mapping", expanded=False):
    st.caption("We guessed these. Change them if needed.")
    options = ["(none)"] + list(raw.columns)
    guesses = {
        "date": guess(raw, ["date", "month", "time", "day"]),
        "sales": guess(raw, ["sales", "revenue", "amount", "total"]),
    }
    guesses["profit"] = guess(raw, ["profit", "margin"], exclude=[guesses["sales"]])
    guesses["product"] = guess(raw, ["product", "item", "sku"])
    guesses["category"] = guess(raw, ["category", "segment", "department", "type"])
    guesses["region"] = guess(raw, ["region", "state", "city", "store", "location", "country"])
    guesses["customer"] = guess(raw, ["customer", "client"])
    guesses["qty"] = guess(raw, ["quantity", "qty", "units"])
    cols = {}
    for k, g in guesses.items():
        pick = st.selectbox(k.title(), options, index=options.index(g) if g else 0, key=f"map_{k}")
        cols[k] = None if pick == "(none)" else pick

if not cols["sales"]:
    st.error("Please choose which column holds the sales amounts (sidebar → Column mapping).")
    st.stop()

df, notes = clean(raw, cols)
if df.empty:
    st.error("No usable rows remain after cleaning. Check the column mapping in the sidebar.")
    st.stop()

# Filters
st.sidebar.markdown("## 🎛️ Filters")
fdf = df
if cols["date"]:
    lo, hi = df[cols["date"]].min().date(), df[cols["date"]].max().date()
    rng = st.sidebar.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
    if isinstance(rng, tuple) and len(rng) == 2:
        fdf = fdf[(fdf[cols["date"]].dt.date >= rng[0]) & (fdf[cols["date"]].dt.date <= rng[1])]
for key in ("product", "category", "region"):
    c = cols[key]
    if c:
        choice = st.sidebar.multiselect(c, sorted(df[c].astype(str).unique()))
        if choice:
            fdf = fdf[fdf[c].astype(str).isin(choice)]
if fdf.empty:
    st.warning("No rows match these filters. Widen the date range or clear a filter.")
    st.stop()

S, P = cols["sales"], cols["profit"]
facts = build_facts(fdf, cols)
tr = facts["trend"]
summary_text = facts_text(facts, cols)

tabs = st.tabs(["📋 Data & Summary", "📈 Trends", "💡 Insights", "🤖 Compare AI",
                "🎯 Recommendations", "📊 Dashboard", "⬇️ Reports"])

# ----------------------------------------------------------------------------------
# TAB 1: DATASET + SUMMARY
# ----------------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Dataset preview")
    st.dataframe(raw.head(5), use_container_width=True)
    c1, c2 = st.columns([1, 2])
    c1.markdown(kpi("Dataset size", f"{raw.shape[0]:,} × {raw.shape[1]}", "rows × columns", "#FFFFFF", LBLUE),
                unsafe_allow_html=True)
    c2.dataframe(pd.DataFrame({"Column": raw.columns, "Type": raw.dtypes.astype(str).values,
                               "Missing": raw.isna().sum().values}), use_container_width=True, height=180)
    if notes:
        box("orange", "🧹 Data cleaning applied", "<br>".join(notes))
    else:
        box("green", "✅ Data looks clean", "No duplicates, missing values or bad entries needed fixing.")

    st.subheader("Business summary")
    k = st.columns(4)
    k[0].markdown(kpi("💰 Total Sales", money(facts["total_sales"]), f"{facts['rows']:,} transactions",
                      "#D8F3DC", "#15803D"), unsafe_allow_html=True)
    k[1].markdown(kpi("📈 Total Profit", money(facts["total_profit"]) if P else "n/a",
                      f"Margin {facts['margin']:.1f}%" if P else "No profit column", "#DBEAFE", LBLUE),
                  unsafe_allow_html=True)
    k[2].markdown(kpi("🧾 Average Sale", money(facts["avg_sales"]), "per record", "#EDE9FE", "#7C3AED"),
                  unsafe_allow_html=True)
    top_p = facts["product"].iloc[0, 0] if "product" in facts else "n/a"
    k[3].markdown(kpi("🏆 Best Product", str(top_p)[:22], "by total sales", "#FFE5B4", "#C2410C"),
                  unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    if "product" in facts:
        b1.markdown("**Best-selling products**")
        b1.dataframe(facts["product"].head(5), use_container_width=True, hide_index=True)
    if "category" in facts:
        b2.markdown("**Top categories**")
        b2.dataframe(facts["category"].head(5), use_container_width=True, hide_index=True)

    if st.button("✨ Write an AI summary in plain English"):
        try:
            with st.spinner("AI is reading your data..."):
                st.session_state["ai_summary"] = ask_ai(
                    "You explain retail data to a store owner with no analytics background. Use only the numbers given.",
                    f"Summarize this sales data in 5 short, simple sentences:\n\n{summary_text}")
        except Exception as e:
            st.error(f"AI summary failed: {e}")
    if "ai_summary" in st.session_state:
        box("purple", "🤖 AI summary", st.session_state["ai_summary"].replace("\n", "<br>"))

# ----------------------------------------------------------------------------------
# TAB 2: TRENDS
# ----------------------------------------------------------------------------------
with tabs[1]:
    if not tr:
        st.info("Trend analysis needs a date column. Select one in the sidebar → Column mapping.")
    else:
        m = tr["monthly"]
        span = (fdf[cols["date"]].max() - fdf[cols["date"]].min()).days
        if tr["direction"] == "growing":
            box("green", "📈 Sales are growing", f"Sales trend upward by about {tr['slope']:.1f}% of the average month, every month.")
        elif tr["direction"] == "declining":
            box("orange", "⚠️ Sales are declining", f"Sales trend downward by about {abs(tr['slope']):.1f}% of the average month, every month.")
        else:
            box("blue", "➡️ Sales are steady", "No clear upward or downward trend over this period.")
        a, b = st.columns(2)
        a.markdown(kpi("🔝 Peak month", tr["peak"]["Month"], f"Sales {money(tr['peak']['Sales'])}", "#D8F3DC", "#15803D"),
                   unsafe_allow_html=True)
        b.markdown(kpi("🔻 Lowest month", tr["low"]["Month"], f"Sales {money(tr['low']['Sales'])}", "#FFE5B4", "#C2410C"),
                   unsafe_allow_html=True)
        fig = px.line(m, x=cols["date"], y="Sales", markers=True, title="Monthly sales", color_discrete_sequence=[BLUE])
        fig.update_layout(xaxis_title="Month", yaxis_title="Sales")
        st.plotly_chart(fig, use_container_width=True)
        g = m.dropna(subset=["Growth %"])
        if not g.empty:
            fig = px.bar(g, x="Month", y="Growth %", title="Month-over-month growth (%)",
                         color=g["Growth %"] > 0, color_discrete_map={True: "#15803D", False: "#C2410C"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            worst = g.loc[g["Growth %"].idxmin()]
            best = g.loc[g["Growth %"].idxmax()]
            box("green", "Biggest jump", f"{best['Month']}: {best['Growth %']:+.1f}% vs the month before.")
            box("orange", "Biggest drop", f"{worst['Month']}: {worst['Growth %']:+.1f}% vs the month before.")
        if span <= 120:
            w = time_series(fdf, cols, "W")
            st.plotly_chart(px.line(w, x=cols["date"], y="Sales", markers=True, title="Weekly sales",
                                    color_discrete_sequence=["#8B5CF6"]), use_container_width=True)
        if span > 365:
            y = time_series(fdf, cols, "YS")
            y["Year"] = y[cols["date"]].dt.year.astype(str)
            st.plotly_chart(px.bar(y, x="Year", y="Sales", title="Yearly sales", color_discrete_sequence=[LBLUE]),
                            use_container_width=True)

# ----------------------------------------------------------------------------------
# TAB 3: INSIGHTS
# ----------------------------------------------------------------------------------
with tabs[2]:
    if "product" in facts:
        g = facts["product"]
        box("green", "🏆 Top performers", "<br>".join(f"{r.iloc[0]} — {money(r['Sales'])}" for _, r in g.head(3).iterrows()))
        box("orange", "📉 Low performers", "<br>".join(f"{r.iloc[0]} — {money(r['Sales'])}" for _, r in g.tail(3).iterrows()))
        share = g["Sales"].head(3).sum() / g["Sales"].sum() * 100
        box("purple", "🔎 Concentration", f"Your top 3 products bring in <b>{share:.0f}%</b> of sales. "
            + ("That is a high dependency on a few items." if share > 60 else "Sales are reasonably spread out."))
    if "category" in facts:
        g = facts["category"]
        box("green", "🗂️ Leading category", f"<b>{g.iloc[0, 0]}</b> contributes {g['Sales'].iloc[0] / g['Sales'].sum() * 100:.0f}% of sales.")
    if P:
        key = "category" if "category" in facts else "product" if "product" in facts else None
        if key:
            g = facts[key].dropna(subset=["Margin %"])
            hi, lo = g.loc[g["Margin %"].idxmax()], g.loc[g["Margin %"].idxmin()]
            box("green", "💎 Highest margin", f"{hi.iloc[0]} earns {hi['Margin %']:.1f}% margin.")
            box("orange", "⚠️ Lowest margin", f"{lo.iloc[0]} earns only {lo['Margin %']:.1f}% margin.")
            opp = g[(g["Margin %"] > g["Margin %"].median()) & (g["Sales"] < g["Sales"].median())]
            if not opp.empty:
                box("blue", "🚀 Sales opportunity", "High margin but low sales — promote these: "
                    + ", ".join(opp.iloc[:, 0].astype(str).head(4)))
    if "customer" in facts:
        g = facts["customer"]
        share = g["Sales"].head(max(1, len(g) // 10)).sum() / g["Sales"].sum() * 100
        box("purple", "🧑‍🤝‍🧑 Customer pattern", f"The top 10% of customers ({max(1, len(g) // 10)}) generate {share:.0f}% of sales. "
            f"Best customer: {g.iloc[0, 0]} ({money(g['Sales'].iloc[0])}).")
        st.plotly_chart(px.bar(g.head(10), x=g.columns[0], y="Sales", title="Top 10 customers",
                               color_discrete_sequence=[BLUE]), use_container_width=True)
    if "region" in facts:
        g = facts["region"]
        box("blue", "🌍 Regional view", f"Strongest: <b>{g.iloc[0, 0]}</b>. Weakest: <b>{g.iloc[-1, 0]}</b>.")
    if not any(k in facts for k in ("product", "category", "customer", "region")) and not P:
        st.info("Map product, category, region, customer or profit columns to unlock more insights.")

# ----------------------------------------------------------------------------------
# TAB 4: COMPARE TWO AI ANALYSES
# ----------------------------------------------------------------------------------
with tabs[3]:
    st.markdown("Two AI 'analysts' study the same data with different styles. We then check both against your real numbers.")
    if st.button("🤖 Generate and compare analyses"):
        try:
            with st.spinner("Running both analyses..."):
                a = ask_ai("You are a careful data analyst. Describe only what the numbers show. "
                           "Use headings: Key Trends, Key Insights, Risks, Recommendations. Quote real names and figures.",
                           f"Analyze this retail dataset:\n\n{summary_text}", 0.2)
                b = ask_ai("You are a growth-focused retail strategist. Focus on opportunities and bold actions, "
                           "but stay grounded in the numbers. Use headings: Key Trends, Key Insights, Opportunities, Recommendations.",
                           f"Analyze this retail dataset:\n\n{summary_text}", 0.8)
                j = ask_ai("You are an impartial reviewer. Compare two analyses against the real data. "
                           "State: 1) where they agree, 2) where they differ, 3) which is more accurate and useful and why. "
                           "Be concise.",
                           f"REAL DATA:\n{summary_text}\n\nANALYSIS A:\n{a}\n\nANALYSIS B:\n{b}", 0.2)
            st.session_state["cmp"] = (a, b, j)
        except Exception as e:
            st.error(f"AI comparison failed: {e}")
    if "cmp" in st.session_state:
        a, b, j = st.session_state["cmp"]
        sa, sb = fact_check(a, facts), fact_check(b, facts)
        c1, c2 = st.columns(2)
        for col, name, txt, sc, key in ((c1, "📘 Analysis A — Data Analyst", a, sa, "cardA"),
                                        (c2, "📗 Analysis B — Growth Strategist", b, sb, "cardB")):
            with col:
                with st.container(key=key):
                    st.markdown(f'<div class="card-title">{name}</div>', unsafe_allow_html=True)
                    if sc is not None:
                        st.progress(sc, text=f"Fact match with your data: {sc * 100:.0f}%")
                    st.markdown(txt)
        with st.container(key="cardJudge"):
            st.markdown('<div class="card-title">⚖️ Verdict: most useful and reliable</div>', unsafe_allow_html=True)
            if sa is not None and sb is not None:
                winner = "A" if sa > sb else "B" if sb > sa else "tie"
                st.markdown(f"**Fact-check result:** {'Analysis ' + winner if winner != 'tie' else 'Both analyses'} "
                            f"{'matches' if winner != 'tie' else 'match'} the real data best "
                            f"(A: {sa * 100:.0f}%, B: {sb * 100:.0f}%).")
            st.markdown(j)

# ----------------------------------------------------------------------------------
# TAB 5: RECOMMENDATIONS
# ----------------------------------------------------------------------------------
with tabs[4]:
    if st.button("🎯 Generate business recommendations"):
        try:
            with st.spinner("AI is building your action plan..."):
                st.session_state["recs"] = ask_ai(
                    "You are a retail consultant. Every recommendation must reference specific products, categories, "
                    "months or numbers from the data.",
                    "Give recommendations under these headings: Top 3 Priorities (most important first), Pricing, "
                    "Inventory Management, Promotions, Marketing. Keep each point to 1-2 sentences.\n\n" + summary_text)
        except Exception as e:
            st.error(f"Could not generate recommendations: {e}")
    if "recs" in st.session_state:
        with st.container(key="cardRec"):
            st.markdown(st.session_state["recs"])

# ----------------------------------------------------------------------------------
# TAB 6: DASHBOARD
# ----------------------------------------------------------------------------------
with tabs[5]:
    st.caption("Charts follow the filters in the sidebar.")
    r1a, r1b = st.columns(2)
    if tr:
        r1a.plotly_chart(px.line(tr["monthly"], x=cols["date"], y="Sales", markers=True, title="Monthly sales",
                                 color_discrete_sequence=[BLUE]), use_container_width=True)
    if "product" in facts:
        g = facts["product"].head(10)
        r1b.plotly_chart(px.bar(g, x="Sales", y=g.columns[0], orientation="h", title="Top 10 products by sales",
                                color_discrete_sequence=[LBLUE]).update_yaxes(autorange="reversed"),
                         use_container_width=True)
    r2a, r2b = st.columns(2)
    if "category" in facts:
        r2a.plotly_chart(px.pie(facts["category"], names=facts["category"].columns[0], values="Sales",
                                title="Sales by category", hole=0.4, color_discrete_sequence=PALETTE),
                         use_container_width=True)
    if P:
        if tr:
            pm = fdf.set_index(cols["date"])[[S, P]].resample("MS").sum().reset_index()
            fig = go.Figure()
            fig.add_bar(x=pm[cols["date"]], y=pm[P], name="Profit", marker_color="#15803D")
            fig.add_scatter(x=pm[cols["date"]], y=pm[P] / pm[S].replace(0, np.nan) * 100, name="Margin %",
                            yaxis="y2", line=dict(color="#F59E0B"))
            fig.update_layout(title="Profit analysis", yaxis2=dict(overlaying="y", side="right", title="Margin %"),
                              legend=dict(orientation="h"))
            r2b.plotly_chart(fig, use_container_width=True)
    comp = next((k for k in ("region", "category", "product") if k in facts), None)
    if comp:
        g = facts[comp].head(10)
        ys = ["Sales"] + (["Profit"] if P else [])
        fig = px.bar(g, x=g.columns[0], y=ys, barmode="group", title=f"Sales vs profit by {comp}",
                     color_discrete_sequence=[BLUE, "#15803D"])
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Filtered data")
    st.dataframe(fdf, use_container_width=True, height=300)

# ----------------------------------------------------------------------------------
# TAB 7: REPORTS
# ----------------------------------------------------------------------------------
with tabs[6]:
    lines = [f"# Retail Sales Analysis Report", f"Generated {datetime.now():%Y-%m-%d %H:%M} from `{upload.name}`", "",
             "## Key numbers", f"- Total sales: {money(facts['total_sales'])}", f"- Average sale: {money(facts['avg_sales'])}"]
    if P:
        lines.append(f"- Total profit: {money(facts['total_profit'])} (margin {facts['margin']:.1f}%)")
    if tr:
        lines += ["", "## Trends", f"- Overall direction: {tr['direction']}",
                  f"- Peak month: {tr['peak']['Month']} ({money(tr['peak']['Sales'])})",
                  f"- Lowest month: {tr['low']['Month']} ({money(tr['low']['Sales'])})"]
    lines += ["", "## Business insights"]
    if "product" in facts:
        lines.append("- Top products: " + ", ".join(facts["product"].iloc[:3, 0].astype(str)))
        lines.append("- Lowest products: " + ", ".join(facts["product"].iloc[-3:, 0].astype(str)))
    if "category" in facts:
        lines.append(f"- Leading category: {facts['category'].iloc[0, 0]}")
    if "ai_summary" in st.session_state:
        lines += ["", "## AI summary", st.session_state["ai_summary"]]
    if "cmp" in st.session_state:
        a, b, j = st.session_state["cmp"]
        lines += ["", "## AI Analysis A (Data Analyst)", a, "", "## AI Analysis B (Growth Strategist)", b,
                  "", "## Comparison verdict", j]
    lines += ["", "## Recommendations", st.session_state.get("recs", "_Generate recommendations in the app to include them here._")]
    report = "\n".join(lines)
    st.markdown("Generate the AI tabs first if you want their results included in the report.")
    d1, d2 = st.columns(2)
    d1.download_button("📄 Download analysis report", report, "sales_report.md", "text/markdown")
    d2.download_button("🧾 Download filtered data (CSV)", fdf.to_csv(index=False), "filtered_sales.csv", "text/csv")
    with st.expander("Preview report"):
        st.markdown(report)
