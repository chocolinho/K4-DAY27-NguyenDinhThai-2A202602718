"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", page_icon="🤖", layout="wide")
st.title("HITL PR Review Agent")


# ─── Helper: fetch recent sessions from SQLite ─────────────────────────────
async def get_recent_sessions():
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MIN(timestamp)        AS started,
                   MAX(timestamp)        AS last_event,
                   MAX(risk_level)       AS worst_risk,
                   COUNT(*)              AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 25
            """
        ) as cur:
            return await cur.fetchall()


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")
    try:
        sessions = asyncio.run(get_recent_sessions())
        if not sessions:
            st.caption("No review sessions found yet.")
        else:
            table_data = [
                {
                    "Thread ID": f"{s['thread_id'][:8]}...",
                    "PR": s["pr_url"].split("github.com/")[-1] if "github.com/" in s["pr_url"] else s["pr_url"],
                    "Risk": s["worst_risk"],
                    "Events": s["events"],
                }
                for s in sessions
            ]
            st.dataframe(table_data, use_container_width=True, hide_index=True)

            thread_options = {s["thread_id"]: f"{s['thread_id'][:8]}... ({s['pr_url'].split('github.com/')[-1]})" for s in sessions}
            selected_tid = st.selectbox(
                "Select session to inspect",
                options=list(thread_options.keys()),
                format_func=lambda x: thread_options[x],
            )
            if st.button("Load Session", use_container_width=True):
                chosen = next(s for s in sessions if s["thread_id"] == selected_tid)
                st.session_state.thread_id = selected_tid
                st.session_state.pr_url = chosen["pr_url"]
                st.session_state.final = None
                st.session_state.interrupt_payload = None
                st.rerun()
    except Exception as e:
        st.caption(f"Could not load sessions: {e}")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review", type="primary")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    comments = payload.get("comments", [])
    if comments:
        st.markdown("**Proposed Review Comments:**")
        for c in comments:
            st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff Preview"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback / Reviewer Note (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary", use_container_width=True):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject", use_container_width=True):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit", use_container_width=True):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("**Risks identified:**\n- " + "\n- ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation_form"):
        st.markdown("### Reviewer Clarification Questions")
        answers: dict[str, str] = {}
        for i, q in enumerate(payload.get("questions", [])):
            answers[q] = st.text_input(f"Q{i+1}: {q}", key=f"esc_q_{i}")
        if st.form_submit_button("Submit answers", type="primary"):
            return answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            result = await app.ainvoke(
                {"pr_url": pr_url, "thread_id": thread_id}, cfg
            )
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)
        return result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("Fetching PR + asking the LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming agent..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    analysis = final.get("analysis")

    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"✓ {action} — Review comment posted successfully!")
    elif action == "rejected":
        st.warning("Review rejected — no comment posted to GitHub.")
    else:
        st.info(f"Final Action: {action}")

    if analysis:
        st.metric("Review Confidence", f"{analysis.confidence:.0%}")
        st.markdown(f"**LLM Summary:** {analysis.summary}")
        if analysis.confidence_reasoning:
            st.caption(f"Reasoning: {analysis.confidence_reasoning}")

    st.markdown(f"[🔗 View Pull Request on GitHub]({st.session_state.pr_url})")
    st.caption(
        f"thread_id = `{st.session_state.thread_id}`  ·  Replay command: "
        f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )
