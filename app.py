"""Streamlit UI: chat (Employee), escalations (HR Rep), uploads + analytics (Admin).
Run:  streamlit run app.py
"""
import pandas as pd
import plotly.express as px
import streamlit as st
from config import DATA_DIR, AUDIT_LOG, ESCALATION_LOG, DISCLAIMER, check_login
from utils import log_feedback, read_log

st.set_page_config(page_title="HR Policy Assistant", page_icon="💼", layout="wide")


# ---------------- login ----------------
def login():
    st.title("💼 HR Policy Assistant")
    with st.form("login"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            role = check_login(u, p)
            if role:
                st.session_state.update(user=u.strip().lower(), role=role, messages=[])
                st.rerun()
            st.error("Invalid credentials.")
    st.caption("Demo users: employee/employee123, hr/hr123, admin/admin123")


# ---------------- chat ----------------
@st.cache_resource(show_spinner=False)
def get_agent_cached():
    from rag import PolicyAgent
    return PolicyAgent()


def chat_page():
    st.subheader("Ask about HR policies")
    st.info(DISCLAIMER)
    try:
        if "agent" not in st.session_state:
            st.session_state.agent = __import__("rag").PolicyAgent()
    except Exception as e:
        st.error(f"Cannot start the assistant: {e}")
        return
    agent = st.session_state.agent

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        agent.memory.clear()
        st.rerun()

    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                if m.get("escalate"):
                    st.warning("⚠️ This conversation has been escalated to an HR representative.")
                if m.get("sources"):
                    with st.expander("📚 Sources"):
                        for s in m["sources"]:
                            st.markdown(f"**[{s['source']}, Page {s['page']}]** - {s['section']}")
                            st.caption(s["text"])
                c1, c2, _ = st.columns([1, 1, 10])
                if c1.button("👍", key=f"up{i}"):
                    log_feedback(user=st.session_state.user, query=m.get("query", ""), rating="up")
                    st.toast("Thanks for the feedback!")
                if c2.button("👎", key=f"dn{i}"):
                    log_feedback(user=st.session_state.user, query=m.get("query", ""), rating="down")
                    st.toast("Feedback recorded.")

    q = st.chat_input("Type your question...")
    if q is not None:
        st.session_state.messages.append({"role": "user", "content": q})
        with st.spinner("Searching HR documents..."):
            r = agent.ask(q, user=st.session_state.user, role=st.session_state.role)
        st.session_state.messages.append({"role": "assistant", "content": r["answer"], "sources": r["sources"],
                                          "escalate": r["escalate"], "query": q})
        st.rerun()


# ---------------- HR rep ----------------
def escalations_page():
    st.subheader("Escalated conversations")
    df = read_log(ESCALATION_LOG)
    if df.empty:
        st.success("No escalations yet.")
        return
    st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)
    for _, r in df.sort_values("timestamp", ascending=False).head(10).iterrows():
        with st.expander(f"{r['timestamp']} - {str(r['question'])[:60]}"):
            st.write("**Reasons:**", r["reasons"])
            st.write("**3-line summary:**")
            st.text(r["summary"])


# ---------------- admin ----------------
def admin_page():
    tab1, tab2 = st.tabs(["📊 Analytics", "📁 Knowledge base"])
    with tab1:
        df = read_log(AUDIT_LOG)
        if df.empty:
            st.info("No queries logged yet.")
        else:
            df["escalated"] = df["escalated"].astype(str).str.lower() == "true"
            contain = 100 * (~df["escalated"] & (df["status"] == "answered")).mean()
            c1, c2, c3 = st.columns(3)
            c1.metric("Containment rate", f"{contain:.1f}%")
            c2.metric("Total queries", len(df))
            c3.metric("Avg response (s)", f"{df['latency_s'].mean():.2f}")

            df["date"] = pd.to_datetime(df["timestamp"]).dt.date
            vol = df.groupby("date").size().reset_index(name="queries")
            st.plotly_chart(px.line(vol, x="date", y="queries", markers=True, title="Query volume trend"),
                            use_container_width=True)

            idk = df[df["status"] == "idk"]["query"].value_counts().head(5).reset_index()
            idk.columns = ["query", "count"]
            st.markdown("#### Top 5 unanswered questions (AI said I_DONT_KNOW)")
            if idk.empty:
                st.caption("None yet.")
            else:
                st.plotly_chart(px.bar(idk, x="count", y="query", orientation="h"), use_container_width=True)

            esc = read_log(ESCALATION_LOG)
            if not esc.empty:
                reasons = esc["reasons"].fillna("").str.split(";").explode().str.strip()
                reasons = reasons[reasons != ""].value_counts().reset_index()
                reasons.columns = ["reason", "count"]
                st.plotly_chart(px.pie(reasons, names="reason", values="count", title="Escalation reasons"),
                                use_container_width=True)
    with tab2:
        up = st.file_uploader("Upload policy PDFs / CSVs", type=["pdf", "csv"], accept_multiple_files=True)
        if up and st.button("Save & re-index"):
            for f in up:
                (DATA_DIR / f.name).write_bytes(f.getbuffer())
            try:
                from ingest import build_index
                with st.spinner("Indexing..."):
                    n = build_index()
                st.session_state.pop("agent", None)
                st.success(f"Indexed {n} chunks.")
            except Exception as e:
                st.error(f"Indexing failed: {e}")
        st.write("Files in knowledge base:", sorted(p.name for p in DATA_DIR.iterdir()))


# ---------------- router ----------------
if "role" not in st.session_state:
    login()
    st.stop()

role = st.session_state.role
st.sidebar.write(f"👤 **{st.session_state.user}** ({role})")
if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

pages = {"Employee": ["Chat"], "HR Rep": ["Chat", "Escalations"], "Admin": ["Chat", "Escalations", "Admin"]}[role]
# Employee: chat only. HR Rep: + escalations. Admin: + uploads/analytics.
choice = st.sidebar.radio("Go to", pages)
{"Chat": chat_page, "Escalations": escalations_page, "Admin": admin_page}[choice]()
