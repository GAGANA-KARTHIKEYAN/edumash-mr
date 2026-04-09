# app.py — eduMASH-MR Personalized AI Tutor Chatbot
# streamlit run app.py

import streamlit as st
import streamlit.components.v1 as components
import os, json, random, time

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="eduMASH-MR · AI Tutor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* ── Fix: Material Symbols font icons render correctly ───────────── */
.material-symbols-rounded {
    font-family: 'Material Symbols Rounded' !important;
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
}
/* Suppress raw icon-name text that renders when font hasn't loaded */
[data-testid="stExpanderToggleIcon"] svg,
[data-testid="stExpanderChevron"] { display: inline-flex !important; }

/* Hide the raw text fallback for any Streamlit icon span */
span[data-testid*="Icon"] { 
    font-size: 0 !important; 
    line-height: 0 !important;
}
span[data-testid*="Icon"]::before {
    font-size: 1rem !important;
    line-height: 1.4 !important;
}

/* ── Fix 1: File uploader "uploadUpload" double text bug ─────────── */
/* Streamlit renders a hidden span + visible button text — hide the ghost */
[data-testid="stFileUploader"] button span:first-child {
    display: none !important;
}
[data-testid="stFileUploader"] button::before {
    content: "Browse files";
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem;
}

/* ── Fix 2: Material icon names showing as raw text ──────────────── */
/* Strip any raw icon-name text leaking into labels */
[data-testid="stWidgetLabel"] p,
[data-testid="stSidebarContent"] label p,
.stTextInput label p,
.stSelectbox label p {
    font-size: 0.875rem !important;
    line-height: 1.4 !important;
    overflow: hidden !important;
}

/* ── Fix 3: RAG context expander label overlap ───────────────────── */
[data-testid="stExpander"] summary p {
    font-size: 0.875rem !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Fix 4: Sidebar icon/label stacking ─────────────────────────── */
[data-testid="stSidebarContent"] .stMarkdown p {
    margin-bottom: 0.25rem !important;
    line-height: 1.5 !important;
}

/* ── Fix 5: Audio recorder label overlap ─────────────────────────── */
[data-testid="stAudioInput"] label {
    display: block !important;
    overflow: visible !important;
    white-space: normal !important;
}

/* Score pill - theme agnostic */
.pill {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 0.8rem; font-weight: 500; margin: 2px;
}
.pill-green  { background: rgba(34, 197, 94, 0.2); border: 1px solid rgba(34, 197, 94, 0.4); }
.pill-amber  { background: rgba(245, 158, 11, 0.2); border: 1px solid rgba(245, 158, 11, 0.4); }
.pill-red    { background: rgba(239, 68, 68, 0.2); border: 1px solid rgba(239, 68, 68, 0.4); }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# Session State Initialization
# ═══════════════════════════════════════════════════════════════════
def _init():
    defaults = {
        "phase"           : "upload",
        "messages"        : [],
        "retriever"       : None,
        "question_count"  : 0,
        "total_questions" : 5,
        "current_question": None,
        "asked_questions" : [],
        "awaiting_answer" : False,
        "student_name"    : "Student",
        "language"        : "English",
        "student_profile" : {
            "scores"           : [],
            "correct_concepts" : [],
            "missing_concepts" : [],
            "answers"          : [],
            "questions"        : [],
        },
        "gemini_ok"       : False,
        "curriculum_summary": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ── Auto-load API keys from Streamlit Cloud secrets ─────────────────
if not st.session_state.get("groq_ok") and not st.session_state.get("gemini_ok"):
    try:
        import explanation_module.engine as engine
        _groq_secret = st.secrets.get("GROQ_API_KEY", "")
        if _groq_secret and engine._groq_client is None:
            if engine.configure_groq(_groq_secret):
                st.session_state.groq_ok = True
                st.session_state.gemini_ok = True
        if engine._groq_client is None:
            _gemini_secret = st.secrets.get("GEMINI_API_KEY", "")
            if _gemini_secret:
                if engine.configure_gemini(_gemini_secret):
                    st.session_state.gemini_ok = True
    except Exception:
        pass  # Secrets not configured — user must enter key manually


# ═══════════════════════════════════════════════════════════════════
# Helper: add a message to the chat history
# ═══════════════════════════════════════════════════════════════════
def add_msg(role: str, content: str, meta: dict = None):
    st.session_state.messages.append({"role": role, "content": content, "meta": meta or {}})


# ═══════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 eduMASH-MR")
    st.caption("Personalized Graph RAG Tutor")
    st.markdown("---")

    # ── AI Provider Keys ───────────────────────────────────────────
    st.markdown("### ⚡ AI Mode")
    
    # Groq (primary — no rate limits on free tier)
    groq_key = st.text_input("🟢 Groq API Key (Recommended)", type="password",
                             placeholder="Get free key at console.groq.com",
                             key="groq_key_input")
    # Check and re-sync engine if key exists but client was lost in worker restart
    import explanation_module.engine as engine
    if groq_key and (engine._groq_client is None or not st.session_state.get("groq_ok")):
        if engine.configure_groq(groq_key):
            st.session_state.groq_ok = True
            st.session_state.gemini_ok = True
            st.success("✅ Groq Active — No rate limits!")
        else:
            st.error("❌ Invalid Groq key")
    elif st.session_state.get("groq_ok"):
        st.success("✅ Groq Active — No rate limits!")

    # Gemini (fallback)
    with st.expander("🔵 Gemini API Key (optional fallback)"):
        api_key = st.text_input("Gemini API Key", type="password",
                                placeholder="Paste Gemini key as backup",
                                key="gemini_key_input")
        if api_key and not st.session_state.gemini_ok:
            from explanation_module.engine import configure_gemini
            if configure_gemini(api_key):
                st.session_state.gemini_ok = True
                st.success("✅ Gemini Active (fallback)")
            else:
                st.error("Invalid key")
        elif st.session_state.gemini_ok and not st.session_state.get("groq_ok"):
            st.success("✅ Gemini Active")

    if not st.session_state.gemini_ok and not st.session_state.get("groq_ok"):
        st.info("ℹ️ Enter a Groq or Gemini key above for AI-powered tutoring")


    # ── Multilingual Mode ──────────────────────────────────────
    _LANGS = ["English", "Spanish", "French", "Hindi", "Telugu", "Kannada", "Chinese", "Arabic", "German", "Japanese"]
    selected_lang = st.selectbox(
        "🌐 Language",
        _LANGS,
        index=_LANGS.index(st.session_state.language) if st.session_state.language in _LANGS else 0
    )
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang

    st.markdown("---")

    # ── Student name ───────────────────────────────────────────
    name = st.text_input("Your Name", value=st.session_state.student_name, key="name_input")
    if name != st.session_state.student_name:
        st.session_state.student_name = name

    st.markdown("---")

    # ── Upload Zone ────────────────────────────────────────────
    st.markdown("### 📚 Upload Curriculum")
    st.caption("PDF, PPTX, or TXT — your notes, textbook chapters, lecture slides")
    uploaded = st.file_uploader(" ", type=["pdf", "txt", "ppt", "pptx"],
                                accept_multiple_files=True,
                                label_visibility="collapsed")
    if uploaded and st.session_state.phase == "upload":
        import shutil
        if os.path.exists("data"):
            shutil.rmtree("data")
        os.makedirs("data", exist_ok=True)
        for f in uploaded:
            with open(f"data/{f.name}", "wb") as out:
                out.write(f.read())
        st.success(f"✅ {len(uploaded)} file(s) saved cleanly")
        if st.button("🚀 Start Tutoring Session", type="primary", use_container_width=True):
            st.session_state.phase = "indexing"
            st.rerun()

    elif st.session_state.phase == "upload":
        st.markdown("↑ Upload files then click **Start Tutoring Session**")
        st.caption("Or click below to use built-in demo curriculum")
        if st.button("▶ Use Demo Curriculum", use_container_width=True):
            import shutil
            if os.path.exists("data"):
                shutil.rmtree("data")
            os.makedirs("data", exist_ok=True)
            with open("data/demo.txt", "w", encoding="utf-8") as out:
                out.write("Photosynthesis is a process used by plants to convert light energy into chemical energy. Plants produce glucose from carbon dioxide and water using sunlight. Chlorophyll is the green pigment that absorbs light. Oxygen is produced as a by-product when water molecules are split.\n\nKinematics is the study of motion without considering forces. Velocity is a vector quantity that refers to the rate of change of displacement. Speed is a scalar quantity and does not include direction. Acceleration is the rate of change of velocity with respect to time.")
            st.session_state.phase = "indexing"
            st.rerun()

    st.markdown("---")

    # ── Live Progress Sidebar ──────────────────────────────────
    if st.session_state.phase in ("quiz", "report", "done"):
        sp = st.session_state.student_profile
        scores = sp["scores"]
        q_done = len(scores)
        q_total = st.session_state.total_questions

        st.markdown("### 📊 Your Progress")
        st.progress(q_done / q_total, text=f"Question {q_done}/{q_total}")

        if scores:
            avg = sum(scores)/len(scores)
            col1, col2 = st.columns(2)
            col1.metric("Avg Score", f"{avg:.0%}")
            col2.metric("Questions", f"{q_done}/{q_total}")

        correct = sp["correct_concepts"]
        missing = sp["missing_concepts"]

        if correct:
            st.markdown("**✅ Strong Areas**")
            for c in correct[-3:]:
                st.markdown(f'<div class="progress-concept"><span>{c}</span>'
                            f'<span style="color:#86efac">✓</span></div>',
                            unsafe_allow_html=True)

        if missing:
            st.markdown("**⚠️ Needs Review**")
            for c in missing[-3:]:
                st.markdown(f'<div class="progress-concept"><span>{c}</span>'
                            f'<span style="color:#fca5a5">↺</span></div>',
                            unsafe_allow_html=True)

    # ── Reset button ────────────────────────────────────────────
    if st.session_state.phase not in ("upload",):
        st.markdown("---")
        if st.button("🔄 New Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ═══════════════════════════════════════════════════════════════════
# Main area header
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding: 20px 0 10px 0;">
  <h1 style="margin:0; font-size: 2rem; background: linear-gradient(90deg,#818cf8,#c084fc); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    🧠 eduMASH-MR Tutor
  </h1>
  <p style="margin:0; color:#64748b; font-size:0.9rem;">
    Multimodal · Multilingual · Misconception-Aware · Graph RAG · Personalized Learning
  </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════
# Upload Phase UI
# ═══════════════════════════════════════════════════════════════════
if st.session_state.phase == "upload":
    st.markdown(f"""
    <div class="upload-hint">
      <div style="font-size: 3rem;">📖</div>
      <h2 style="color:#818cf8;">Welcome, {st.session_state.student_name}!</h2>
      <p style="font-size: 1.05rem; color: #94a3b8;">
        Upload your <strong style="color:#a5b4fc">lecture notes, textbook chapters, or PDF slides</strong>
        in the sidebar.<br>
        I'll read them, explain what I found, then quiz you with adaptive questions
        and give you a <strong style="color:#a5b4fc">personalized learning report</strong>.
      </p>
      <p style="color:#6366f1; font-size: 0.9rem;">⬅ Upload files in the sidebar to begin</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**🔍 Graph RAG**\n\nBuilds knowledge graph from your material for concept-aware retrieval.")
    with col2:
        st.success("**🎯 Adaptive Questions**\n\nQuestions target YOUR weak areas, not just random quiz.")
    with col3:
        st.warning("**📊 Personal Report**\n\nGet a detailed mastery report and personalized study plan.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════
# Indexing Phase — Build RAG + generate curriculum summary
# ═══════════════════════════════════════════════════════════════════
if st.session_state.phase == "indexing":
    with st.spinner("🔧 Building Graph RAG index from your curriculum…"):
        from core import build_system
        try:
            retriever = build_system()
            st.session_state.retriever = retriever
        except ValueError as e:
            st.error(f"**Error Indexing Material:** {str(e)}")
            if st.button("⬅️ Go Back to Upload"):
                st.session_state.phase = "upload"
                st.rerun()
            st.stop()

    # ── Render curriculum knowledge graph ──
    with st.spinner("🔗 Constructing interactive knowledge graph…"):
        try:
            from utils.graph_renderer import render_knowledge_graph_html
            kg = retriever.knowledge_graph
            if kg and kg.number_of_nodes() > 0:
                kg_html = render_knowledge_graph_html(kg, title="Curriculum Knowledge Graph", height=480)
                st.session_state.knowledge_graph_html = kg_html
            else:
                st.session_state.knowledge_graph_html = None
        except Exception as e:
            print(f"[app] Graph render failed: {e}")
            st.session_state.knowledge_graph_html = None

    with st.spinner("📖 Analysing curriculum topics…"):
        import explanation_module.engine as engine
        # Force sync before deep analysis to avoid "offline" status 
        if st.session_state.get("groq_key_input") and engine._groq_client is None:
            engine.configure_groq(st.session_state.groq_key_input)
            
        summary = engine.generate_curriculum_summary(retriever, st.session_state.language)
        st.session_state.curriculum_summary = summary
        
        # UI fix: if summary came back as offline but we expected it to be online, warns user
        if summary.get("greeting", "").startswith("⚠️ [OFFLINE"):
            if st.session_state.get("groq_ok"):
                st.warning("⚠️ AI Analysis timed out or failed to parse. Using offline summary (Questions will still attempt AI analysis).")

    # Craft intro messages
    s = summary
    topics_str = "  \n".join([f"• {t}" for t in s.get("topics", ["General Knowledge"])])

    add_msg("assistant", f"""👋 **Hello, {st.session_state.student_name}!**

{s.get("greeting", "I've read your curriculum!")}

**Topics I found in your material:**
{topics_str}

I identified **{s.get("concept_count", "several")} concepts** in your curriculum's knowledge graph.

{s.get("encouragement", "Let's get started!")}

---
I'll ask you **{st.session_state.total_questions} adaptive questions**, each targeting different concepts. After each answer, I'll tell you exactly what you got right, what's missing, and show you the correct explanation from *your* curriculum.

**Type your answer in the box below. When you're ready, I'll start with Question 1!** 🎯""",
        meta={"type": "knowledge_graph"})

    st.session_state.phase = "quiz"
    st.rerun()


# ═══════════════════════════════════════════════════════════════════
# Render all messages
# ═══════════════════════════════════════════════════════════════════
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧠" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"], unsafe_allow_html=True)

        # Render any rich metadata cards attached to bot messages
        meta = msg.get("meta", {})

        # ── Knowledge Graph (shown after indexing) ──
        if meta.get("type") == "knowledge_graph":
            kg_html = st.session_state.get("knowledge_graph_html")
            if kg_html:
                with st.expander("🔗 Interactive Curriculum Knowledge Graph", expanded=True):
                    components.html(kg_html, height=520, scrolling=True)
            else:
                st.info("ℹ️ Knowledge graph is empty — upload more content for richer concept extraction.")

        elif meta.get("type") == "evaluation":
            ev = meta["eval"]
            c1, c2 = st.columns(2)
            score = ev.get("score", 0)
            score_color = "green" if score > 0.7 else ("amber" if score > 0.4 else "red")
            score_pct = f"{int(score*100)}%"

            with c1:
                st.success(f"**✅ What you got right**\n\n{ev.get('what_student_got_right','—')}")
                st.error(f"**⚠️ What's missing / incorrect**\n\n{ev.get('what_is_missing','—')}")
            with c2:
                st.info(f"**📖 Correct Explanation (from your curriculum)**\n\n{ev.get('correction','—')}")

            missing = ev.get("missing_concepts", [])
            if missing:
                pills = " ".join([f'<span class="pill pill-red">{m}</span>' for m in missing[:6]])
                st.markdown(f"**Concepts to review:** {pills}", unsafe_allow_html=True)

            graph_ctx = ev.get("graph_ctx", "")
            if graph_ctx and graph_ctx.strip():
                with st.expander("🔍 Graph RAG Context (concept relationships retrieved)", expanded=False):
                    st.code(graph_ctx, language="")

            # ── Misconception Comparison Graph ──
            stu_tri = ev.get("student_triplets", [])
            ref_tri = ev.get("ref_triplets", [])
            if stu_tri or ref_tri:
                try:
                    from utils.graph_renderer import render_comparison_graph_html
                    cmp_html = render_comparison_graph_html(stu_tri, ref_tri, height=420)
                    with st.expander("🧠 Misconception Graph — Student vs Reference", expanded=True):
                        components.html(cmp_html, height=460, scrolling=True)
                        st.caption("🟢 Correct concept · 🔴 Missing concept · 🟡 Hallucinated · Solid=correct link · Dashed=missing/wrong")
                except Exception as e:
                    st.warning(f"Graph render error: {e}")

        elif meta.get("type") == "report":
            rpt = meta["report"]
            scores = rpt.get("scores", [])
            avg = rpt.get("avg_score", 0)
            level = rpt.get("mastery_level", "Developing")

            # Score bar chart
            if scores:
                import pandas as pd
                df = pd.DataFrame({
                    "Question": [f"Q{i+1}" for i in range(len(scores))],
                    "Score": [s * 100 for s in scores]
                })
                st.bar_chart(df.set_index("Question"), color="#6366f1")

            c1, c2, c3 = st.columns(3)
            c1.metric("Overall Score", f"{avg:.0%}")
            c2.metric("Mastery Level", level)
            c3.metric("Questions Done", len(scores))

            study_plan = rpt.get("study_plan", [])
            if study_plan:
                st.markdown("**📋 Your Personalized Study Plan:**")
                for i, step in enumerate(study_plan, 1):
                    st.markdown(f"{i}. {step}")

            st.markdown(f"""
            <div class="report-card" style="margin-top:12px">
              <b>💪 {rpt.get('motivation','Keep going!')}</b>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# Quiz Phase — Generate questions and process answers
# ═══════════════════════════════════════════════════════════════════
if st.session_state.phase == "quiz":
    from explanation_module.engine import generate_next_question, evaluate_student_answer_full

    retriever = st.session_state.retriever


    # Multimodal Input Options
    # Multimodal Input Options
    st.markdown("### 🗣️ Multimodal Answer Options")
    col1, col2 = st.columns(2)
    
    with col1:
        img_upload = st.file_uploader("📷 Upload handwritten notes (Image)", type=["png", "jpg", "jpeg"], key="img_upload")
    
    with col2:
        tab_mic, tab_up = st.tabs(["🎙️ Record Voice", "📁 Upload Audio File"])
        with tab_mic:
            audio_record = st.audio_input("Speak your answer natively", key="audio_record")
        with tab_up:
            audio_upload = st.file_uploader("Upload pre-recorded audio", type=["wav", "mp3", "m4a"], key="audio_upload")

    # Final answer parsing
    from input_module.input_handler import get_input

    # Check for text chat input
    chat_val = st.chat_input(
        f"Answer Q{st.session_state.question_count + 1}/{st.session_state.total_questions} — Type, upload image, or upload/record audio!"
    )
    
    # Check for explicit 'Submit Media' intent (to prevent auto-fire on just opening file dialog)
    media_submit = False
    active_audio = audio_record if audio_record else audio_upload
    if (img_upload or active_audio) and st.button("🚀 Submit Media Answer", type="primary"):
        media_submit = True

    if (chat_val or media_submit) and st.session_state.awaiting_answer:
        question = st.session_state.current_question
        
        # Save temp files for input_handler
        img_path = None
        audio_path = None
        
        if img_upload:
            img_path = f"data/temp_{img_upload.name}"
            with open(img_path, "wb") as f:
                f.write(img_upload.getbuffer())
        
        if active_audio:
            audio_path = f"data/temp_audio.wav"
            with open(audio_path, "wb") as f:
                f.write(active_audio.getbuffer())

        with st.spinner("🎙️📷 Processing multimodal input..."):
            answer = get_input(text=chat_val, audio_path=audio_path, image_path=img_path)
            
        if not answer:
            st.warning("⚠️ Could not extract text from your input. Please try again.")
            st.stop()

        # ── Minimum answer quality guard ──────────────────────────────
        _trivial = {"idk", "i don't know", "i dont know", "no idea", "dunno", 
                    "nothing", "not sure", "?", "...", "n/a", "na", "skip"}
        if len(answer.strip()) < 15 or answer.strip().lower() in _trivial:
            st.warning("✏️ Please write at least a sentence explaining your understanding — even a partial answer helps you learn!")
            st.stop()
            
        # Clean up temp files
        if img_path and os.path.exists(img_path): os.remove(img_path)
        if audio_path and os.path.exists(audio_path): os.remove(audio_path)

        add_msg("user", answer)

        with st.spinner("🔍 Analysing your answer with Graph RAG…"):
            ev = evaluate_student_answer_full(question, answer, retriever, st.session_state.language)

        score = ev.get("score", 0)
        score_pct = f"{int(score*100)}%"
        q_num = st.session_state.question_count + 1

        # Score line
        color = "green" if score > 0.7 else ("amber" if score > 0.4 else "red")
        pill_class = f"pill-{color}"
        verdict = "✅ Correct!" if score > 0.6 else ("⚠️ Partially correct" if score > 0.35 else "❌ Needs more work")

        bot_msg = (
            f"**Score: {score_pct}** — {verdict}\n\n"
            f"{ev.get('encouragement','Keep going!')}"
        )

        # Add the tip if available
        tip = ev.get("followup_tip","")
        if tip:
            bot_msg += f"\n\n💡 *{tip}*"

        next_qn = q_num + 1
        if next_qn <= st.session_state.total_questions:
            bot_msg += f"\n\n---\nMoving to **Question {next_qn}**…"
        else:
            bot_msg += "\n\n---\n🎉 **That's all the questions!** Generating your personalized report…"

        add_msg("assistant", bot_msg, meta={"type": "evaluation", "eval": ev})

        # Update student profile
        sp = st.session_state.student_profile
        sp["scores"].append(score)
        sp["questions"].append(question)
        sp["answers"].append(answer)

        if score > 0.6:
            sp["correct_concepts"].append(question.get("concept", ""))
        else:
            missing = ev.get("missing_concepts", [])
            sp["missing_concepts"].extend(missing)
            if question.get("concept"):
                sp["missing_concepts"].append(question["concept"])

        # Advance
        st.session_state.question_count = q_num
        st.session_state.awaiting_answer = False

        if q_num >= st.session_state.total_questions:
            st.session_state.phase = "report"

        st.session_state.current_question = None # Clear after evaluation
        st.rerun()

    # ── Auto-generate next question if not currently waiting for answer ──
    if not st.session_state.awaiting_answer:
        from explanation_module.engine import generate_next_question
        qn = st.session_state.question_count + 1

        if qn > st.session_state.total_questions:
            st.session_state.phase = "report"
            st.rerun()

        with st.spinner(f"🤔 Preparing Question {qn}/{st.session_state.total_questions}…"):
            question = generate_next_question(
                retriever,
                st.session_state.student_profile,
                st.session_state.asked_questions,
                qn,
                st.session_state.language
            )

        st.session_state.current_question = question
        st.session_state.asked_questions.append(question)

        difficulty_pill = {
            "easy"  : '<span class="pill pill-green">Easy</span>',
            "medium": '<span class="pill pill-amber">Medium</span>',
            "hard"  : '<span class="pill pill-red">Hard</span>',
        }.get(question.get("difficulty", "medium"), "")

        concept_pill = f'<span class="pill pill-blue">📌 {question.get("concept","concept")}</span>'

        add_msg("assistant",
            f"**Question {qn}/{st.session_state.total_questions}** "
            f"{difficulty_pill} {concept_pill}\n\n"
            f"**{question['question']}**\n\n"
            f"*Take your time — answer in your own words.*",
        )
        st.session_state.awaiting_answer = True
        st.rerun()

# ═══════════════════════════════════════════════════════════════════
# Report Phase
# ═══════════════════════════════════════════════════════════════════
elif st.session_state.phase == "report":
    from explanation_module.engine import generate_personalized_report

    retriever = st.session_state.retriever
    sp = st.session_state.student_profile

    with st.spinner("📊 Generating your personalized learning report…"):
        report = generate_personalized_report(sp, retriever, st.session_state.language)

    avg = report.get("avg_score", 0)
    level = report.get("mastery_level", "Developing")

    report_msg = f"""
---
## 🎓 Your Personalized Learning Report

### {report.get('headline', 'Great effort!')}

**💪 Strengths:**
{report.get('strength_summary','—')}

**🔧 Areas to Improve:**
{report.get('weakness_summary','—')}

---

> **Mastery Level: {level}** | **Overall Score: {avg:.0%}**
"""
    add_msg("assistant", report_msg, meta={"type": "report", "report": report})

    st.session_state.phase = "done"
    st.rerun()


# ═══════════════════════════════════════════════════════════════════
# Done Phase — offer retry
# ═══════════════════════════════════════════════════════════════════
elif st.session_state.phase == "done":
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Retake Quiz (same material)", use_container_width=True, type="primary"):
            # Keep retriever, reset quiz state
            r = st.session_state.retriever
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            _init()
            st.session_state.retriever = r
            st.session_state.phase = "quiz"
            st.rerun()
    with col2:
        if st.button("📤 Upload New Material", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
