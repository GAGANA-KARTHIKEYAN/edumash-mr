# explanation_module/engine.py
# Dual-provider: Groq (primary) OR Gemini (fallback) OR offline rule-based

from typing import Dict, List
import os, json, re, random

# ── LLM Provider State ───────────────────────────────────────────────
_groq_client  = None
_gemini_model = None
_active_provider = None   # 'groq' | 'gemini' | None
_groq_model_override = None  # Persists fallback model across all calls in session


def configure_groq(api_key: str) -> bool:
    global _groq_client, _active_provider
    try:
        from groq import Groq
        if not api_key or len(api_key.strip()) < 10:
            return False
        _groq_client = Groq(api_key=api_key.strip())
        _active_provider = "groq"
        print("[engine] Groq configured ✓ (llama-3.3-70b-versatile)")
        return True
    except Exception as e:
        print(f"[engine] Groq config failed: {e}")
        return False


def configure_gemini(api_key: str) -> bool:
    global _gemini_model, _active_provider
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        if _active_provider is None:
            _active_provider = "gemini"
        print("[engine] Gemini configured ✓")
        return True
    except Exception as e:
        print(f"[engine] Gemini config failed: {e}")
        return False


def _llm(prompt: str, retries: int = 2) -> str | None:
    """Call Groq first (with persistent model memory), fall back to Gemini, then offline."""
    global _groq_model_override
    import time

    # ── Try Groq ────────────────────────────────────────────────────
    if _groq_client is not None:
        # Use the session-persisted fallback model if 70B was already rate-limited this session
        current_model = _groq_model_override or "llama-3.3-70b-versatile"
        for attempt in range(retries + 1):
            try:
                resp = _groq_client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=8192,
                    temperature=0.4,
                )
                text = resp.choices[0].message.content.strip()
                with open("engine_debug.log", "a", encoding="utf-8") as f:
                    f.write(f"[Groq] Success attempt={attempt} model={current_model}: {text[:80]}\n")
                return text
            except Exception as e:
                err = str(e)
                with open("engine_debug.log", "a", encoding="utf-8") as f:
                    f.write(f"[Groq] attempt {attempt} failed: {err}\n")
                if "rate" in err.lower() or "429" in err:
                    if current_model != "llama-3.1-8b-instant":
                        print(f"[engine] Groq 70B rate-limited. Permanently switching to 8b-instant for this session...")
                        current_model = "llama-3.1-8b-instant"
                        _groq_model_override = "llama-3.1-8b-instant"  # persist for all future calls
                        time.sleep(0.5)
                    else:
                        print(f"[engine] Groq 8B also rate-limited. Trying Gemini...")
                        break  # both models exhausted, skip to Gemini
                else:
                    break   # non-rate error, skip to Gemini

    # ── Try Gemini fallback ─────────────────────────────────────────
    if _gemini_model is not None:
        for attempt in range(retries):
            try:
                resp = _gemini_model.generate_content(prompt)
                with open("engine_debug.log", "a", encoding="utf-8") as f:
                    f.write(f"[Gemini] Success attempt={attempt}: {resp.text[:80]}\n")
                return resp.text.strip()
            except Exception as e:
                err = str(e)
                with open("engine_debug.log", "a", encoding="utf-8") as f:
                    f.write(f"[Gemini] attempt {attempt} failed: {err}\n")
                if "429" in err or "quota" in err.lower():
                    wait = 20 * (attempt + 1)
                    import streamlit as st
                    st.warning(f"⏳ Gemini quota hit. Waiting {wait}s... (use Groq to avoid this!)")
                    time.sleep(wait)
                else:
                    break

    with open("engine_debug.log", "a", encoding="utf-8") as f:
        f.write("[engine] Both providers failed. Offline fallback.\n")
    return None

def _parse_json(raw: str) -> dict | None:
    """Robustly extract JSON from LLM output, stripping markdown/conversational padding."""
    try:
        # Find exactly the outermost JSON brackets to strip conversational padding
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            clean_json = match.group(0)
            return json.loads(clean_json)
        return json.loads(raw)
    except Exception as e:
        print(f"[engine] JSON Parse Failed. Reason: {e}")
        with open("engine_debug.log", "a", encoding="utf-8") as f:
            f.write(f"[engine] JSON Parse Failed. Raw: {raw[:200]}\n")
        return None

# ═══════════════════════════════════════════════════════════════════
# 1. Curriculum Summary — called once after upload
# ═══════════════════════════════════════════════════════════════════
def generate_curriculum_summary(retriever, language: str = "English") -> str:
    """
    Produce a rich summary of the uploaded curriculum so the bot can
    introduce the topics before quizzing begins.
    """
    # Collect text from graph nodes + top chunks
    graph_nodes = list(retriever.knowledge_graph.nodes())[:30] if retriever.knowledge_graph else []
    sample_chunks = [c.page_content for c in retriever.chunks[:5]]
    doc_ctx = " ".join(sample_chunks)[:1200]
    concept_list = ", ".join(graph_nodes[:20]) if graph_nodes else "various topics"

    prompt = f"""
You are an expert teacher. A student just uploaded their study material.
Summarize what topics are covered in this curriculum in a friendly, encouraging way.
Then list the main concepts the student should know.

CURRICULUM EXCERPT:
{doc_ctx}

KEY CONCEPTS FROM KNOWLEDGE GRAPH:
{concept_list}

Respond in this JSON format:
{{
  "greeting": "A warm 2-3 sentence introduction telling the student what topics you've found in their material.",
  "topics": ["Topic 1", "Topic 2", "Topic 3"],
}}
IMPORTANT: You MUST write your ENTIRE response text (except JSON syntax keys) in {language}!
Only output the JSON. No markdown fences.
"""
    raw = _llm(prompt)
    if raw:
        parsed = _parse_json(raw)
        if parsed:
            return parsed

    # ── Offline fallback ─────────────────────────────────────────
    topic_hints = [n.title() for n in graph_nodes[:6]] if graph_nodes else ["the uploaded material"]
    return {
        "greeting": f"⚠️ [OFFLINE MODE] I've finished reading your curriculum! I found content covering {', '.join(topic_hints[:3])} and more. (Please enter your Gemini API key in the sidebar to enable AI analysis and Language features!)",
        "topics": topic_hints[:5] if topic_hints else ["General Concepts"],
        "concept_count": len(graph_nodes),
        "encouragement": "Let's test your understanding with a few targeted questions. You've got this! 💪"
    }


# ═══════════════════════════════════════════════════════════════════
# 2. Adaptive Question Generator
# ═══════════════════════════════════════════════════════════════════
QUESTION_TEMPLATES = [
    "Explain the concept of **{concept}** in your own words.",
    "What is the relationship between **{a}** and **{b}**?",
    "How does **{concept}** differ from **{other}**?",
    "What happens when **{concept}** is applied? Give an example.",
    "Why is **{concept}** important? What would happen without it?",
    "Can you describe the process of **{concept}** step by step?",
    "What are the key properties of **{concept}**?",
]

def generate_next_question(
    retriever,
    student_profile: dict,
    asked_questions: list,
    question_number: int,
    language: str = "English"
) -> dict:
    """
    Generate the next adaptive question targeting the student's weak areas.
    Returns dict: {question, concept, context, difficulty}
    """
    # Determine target concept (weak first, then new)
    weak = student_profile.get("missing_concepts", [])
    graph_nodes = list(retriever.knowledge_graph.nodes()) if retriever.knowledge_graph else []
    asked_concepts = [q.get("concept", "") for q in asked_questions]

    # Priority: weak unasked → any unasked → random
    candidates = [n for n in weak if n not in asked_concepts]
    if not candidates:
        candidates = [n for n in graph_nodes if n not in asked_concepts]
    if not candidates:
        candidates = graph_nodes or ["the topic"]

    target_concept = random.choice(candidates[:5])
    flat_chunks, graph_ctx, _ = retriever.retrieve(target_concept, k=2)
    context = " ".join(flat_chunks)[:600]

    # LLM question generation
    prompt = f"""
You are a Socratic educational tutor. Generate ONE clear, thought-provoking question
for a student about the concept "{target_concept}".

Curriculum context: {context}

Student's weak areas so far: {", ".join(weak) or "none yet"}
Question number: {question_number} of 5

Rules:
- Do NOT give the answer in the question
- Ask for explanation/understanding, not just yes/no
- Be conversational and encouraging
- Keep it to 1-2 sentences

Respond as JSON:
{{
  "question": "the question text",
  "concept": "{target_concept}",
  "difficulty": "easy|medium|hard",
}}
IMPORTANT: You MUST write the "question" and "hint_if_stuck" strictly in {language}! The concept can stay in English if it's a technical term.
Only output JSON, no fences.
"""
    raw = _llm(prompt)
    if raw:
        parsed = _parse_json(raw)
        if parsed:
            parsed["context"] = context
            parsed["graph_ctx"] = graph_ctx
            return parsed

    # ── Offline fallback ─────────────────────────────────────────
    nodes = list(retriever.knowledge_graph.nodes()) if retriever.knowledge_graph else ["concept"]
    other = random.choice([n for n in nodes if n != target_concept] or ["related concept"])
    template = random.choice(QUESTION_TEMPLATES)
    q_text = template.format(concept=target_concept, a=target_concept, b=other, other=other)

    return {
        "question"   : q_text,
        "concept"    : target_concept,
        "difficulty" : "medium",
        "hint_if_stuck": f"Think about how {target_concept} is defined in your study material.",
        "context"    : context,
        "graph_ctx"  : graph_ctx,
    }


# ═══════════════════════════════════════════════════════════════════
# 3. Full Answer Evaluation
# ═══════════════════════════════════════════════════════════════════
def evaluate_student_answer_full(
    question: dict,
    student_answer: str,
    retriever,
    language: str = "English",
) -> dict:
    """
    Evaluate student's answer against curriculum reference.
    Returns: score [0-1], feedback, missing_concepts, correction, graph_missing
    """
    from misconception_module.detector import detect_misconceptions

    flat_chunks, graph_ctx, seed_nodes = retriever.retrieve(question["concept"], k=3)
    ref_text = " ".join(flat_chunks)

    misc_report = detect_misconceptions(student_answer, ref_text, seed_nodes)
    score = misc_report["score"]
    missing = misc_report["missing_concepts"]
    graph_missing = misc_report.get("graph_missing", [])
    missing_links = misc_report.get("missing_links", [])
    wrong_connections = misc_report.get("incorrect_relations", [])

    # LLM evaluation — plain text delimiters (immune to JSON truncation)
    misconception_summary = ""
    if missing:
        misconception_summary += f"Missing concepts: {', '.join(missing)}. "
    if missing_links:
        misconception_summary += f"Missing relationships: {', '.join(missing_links)}. "
    if wrong_connections:
        misconception_summary += f"Incorrect/hallucinated connections: {', '.join(wrong_connections)}. "

    prompt = f"""You are a rigorous, deeply empathetic educational tutor giving a DETAILED university-level evaluation.

QUESTION ASKED: {question["question"]}
STUDENT'S ANSWER: {student_answer}
CURRICULUM REFERENCE (Ground Truth): {ref_text[:1200]}

STRUCTURAL ANALYSIS FROM AI MODEL:
{misconception_summary if misconception_summary else "No major structural gaps detected."}
BASE SEMANTIC MATCH SCORE: {score:.2f} (0.0 means completely wrong/irrelevant, 1.0 means perfect. Anchor your evaluation around this score.)

LANGUAGE FOR YOUR RESPONSE: {language}

Your task: Write a detailed, university-grade evaluation. Do NOT hallucinate praise if the student's answer is trivial or completely wrong. If the base semantic score is low (<0.3), be strict.

Format your response using EXACTLY these markers (do not change the markers):

[SCORE]
<a float between 0.0 and 1.0 only based strictly on accuracy, e.g. 0.15 for bad answers, 0.95 for excellent>

[WHAT YOU GOT RIGHT]
<If the answer is completely wrong or trivial, just write "Nothing was correct." Otherwise, write 1-2 paragraphs celebrating exactly what the student understood correctly.>

[MISCONCEPTION IDENTIFIED]
<First, state the exact name of the misconception or gap in 1 sentence. Then write 3-4 paragraphs deeply explaining: (1) what the specific misconception IS, (2) WHY the student might have formed this misunderstanding (cognitive/conceptual root cause), (3) what incorrect mental model they are using, (4) how this misconception differs from the actual correct understanding.>

[CORRECT EXPLANATION]
<Write 4-5 long, comprehensive paragraphs giving the complete, correct explanation of the concept from scratch, referencing the curriculum, with formulas/examples/comparisons where relevant. This must be a standalone 1-page explanation a student could study from.>

[FOLLOW UP QUESTION]
<One deep, Socratic follow-up question to extend thinking.>

CRITICAL FORMATTING RULE: The marker labels [SCORE], [WHAT YOU GOT RIGHT], [MISCONCEPTION IDENTIFIED], [CORRECT EXPLANATION], [FOLLOW UP QUESTION] MUST remain exactly in English. Do NOT translate them. Only the content inside each section should be in {language}. Be thorough. Do not skip any section.
"""
    raw = _llm(prompt)
    if raw:
        def _extract(marker_start, marker_end, text):
            """Extract text between two markers, tolerating missing brackets or markdown bold asterisks from the 8b model."""
            # Match optional [, optional **, the marker, optional **, optional ], optional :
            start_p = rf"\[?\*{{0,2}}{re.escape(marker_start)}\*{{0,2}}\]?\:?"
            end_p = rf"\[?\*{{0,2}}{re.escape(marker_end)}\*{{0,2}}\]?\:?"
            pattern = rf'{start_p}\s*(.*?)\s*(?={end_p}|\Z)'
            m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ""

        score_raw   = _extract("SCORE", "WHAT YOU GOT RIGHT", raw)
        praise      = _extract("WHAT YOU GOT RIGHT", "MISCONCEPTION IDENTIFIED", raw)
        misconception = _extract("MISCONCEPTION IDENTIFIED", "CORRECT EXPLANATION", raw)
        correction  = _extract("CORRECT EXPLANATION", "FOLLOW UP QUESTION", raw)
        followup    = _extract("FOLLOW UP QUESTION", "ZZEND", raw)

        # Parse score safely
        try:
            llm_score = float(re.search(r'\d+\.?\d*', score_raw).group())
            llm_score = max(0.0, min(1.0, llm_score))
        except Exception:
            llm_score = score

        # If formal section parsing fails completely on a smaller model, salvage the raw generation context
        if not praise and not correction:
            praise_clean = raw.replace("[SCORE]", "").replace(score_raw, "").strip()
            praise = praise_clean if len(praise_clean) > 20 else "Good attempt!"
            correction = "AI generated unstructured feedback; see above."
            
        what_missing = misconception if misconception else (
            f"Missing concepts: {', '.join(missing)}" if missing else "See the correct explanation below."
        )

        return {
            "score"               : llm_score,
            "is_correct"          : llm_score > 0.6,
            "what_student_got_right": praise,
            "what_is_missing"     : what_missing,
            "correction"          : correction,
                "followup_tip"        : followup or f"Re-read the section on {question['concept']}.",
                "encouragement"       : "Keep going — every answer makes you stronger! 🌱",
                "missing_concepts"    : missing,
                "graph_missing"       : graph_missing,
                "missing_links"       : missing_links,
                "wrong_connections"   : wrong_connections,
                "graph_ctx"           : graph_ctx,
                "student_triplets"    : misc_report.get("student_triplets", []),
                "ref_triplets"        : misc_report.get("ref_triplets", []),
            }

    # ── Offline fallback ─────────────────────────────────────────
    if score > 0.7:
        label = "correct"
        praise = "You captured the key ideas well!"
    elif score > 0.4:
        label = "partial"
        praise = "You have a partial understanding — good start!"
    else:
        label = "needs work"
        praise = "This concept needs more attention."

    correction_base = flat_chunks[0][:400] if flat_chunks else "See your study material."

    missing_str = f"Missing: {', '.join(missing) or 'key details'}"
    if missing_links:
        missing_str += f". Also missing relationships: {', '.join(missing_links)}"
    if wrong_connections:
        missing_str += f". Warning! Incorrect relations: {', '.join(wrong_connections)}"

    return {
        "score"              : score,
        "is_correct"         : score > 0.6,
        "what_student_got_right": praise + "\n\n*(⚠️ You are in Offline Fallback Mode. To receive the 1-Page comprehensive AI explanation in your selected language, please enter your Gemini API Key in the left sidebar!)*",
        "what_is_missing"    : missing_str,
        "correction"         : correction_base,
        "followup_tip"       : f"Re-read the section on {question['concept']} in your notes.",
        "encouragement"      : "Keep going — you're building your understanding! 🌱",
        "missing_concepts"   : missing,
        "graph_missing"      : graph_missing,
        "missing_links"      : missing_links,
        "wrong_connections"  : wrong_connections,
        "graph_ctx"          : graph_ctx,
        "student_triplets"   : misc_report.get("student_triplets", []),
        "ref_triplets"       : misc_report.get("ref_triplets", []),
    }


# ═══════════════════════════════════════════════════════════════════
# 4. Personalized Learning Report
# ═══════════════════════════════════════════════════════════════════
def generate_personalized_report(student_profile: dict, retriever, language: str = "English") -> dict:
    """
    Final personalized report after all quiz questions.
    """
    scores   = student_profile.get("scores", [])
    avg_score = sum(scores) / len(scores) if scores else 0
    missing  = list(set(student_profile.get("missing_concepts", [])))[:8]
    correct  = list(set(student_profile.get("correct_concepts", [])))[:8]
    questions = student_profile.get("questions", [])
    answers   = student_profile.get("answers", [])

    q_summary = "\n".join([f"Q{i+1}: {q.get('concept','?')} — score {s:.0%}"
                            for i, (q, s) in enumerate(zip(questions, scores))])

    prompt = f"""
You are a personalized AI tutor delivering a final learning report to a student.

PERFORMANCE SUMMARY:
{q_summary}

Average Score: {avg_score:.0%}
Concepts the student knows well: {", ".join(correct) or "none yet"}
Concepts the student is weak on: {", ".join(missing) or "none"}

Generate a warm, detailed personalized report. Respond as JSON:
{{
  "headline": "one-line overall assessment (encouraging)",
  "strength_summary": "What they are good at (2 sentences)",
  "weakness_summary": "What needs work (2 sentences)",
  "study_plan": ["Step 1 to improve", "Step 2", "Step 3"],
}}
IMPORTANT: You MUST write your ENTIRE response text (except JSON syntax keys) in {language}!
Only output JSON, no fences.
"""
    raw = _llm(prompt)
    if raw:
        parsed = _parse_json(raw)
        if parsed:
            parsed["avg_score"] = avg_score
            parsed["scores"]    = scores
            return parsed

    # ── Offline fallback ─────────────────────────────────────────
    if avg_score > 0.75:   level = "Proficient"
    elif avg_score > 0.5:  level = "Developing"
    else:                  level = "Beginner"

    return {
        "headline"        : f"You scored {avg_score:.0%} overall — {level}!",
        "strength_summary": f"You demonstrated good command of: {', '.join(correct) or 'several areas'}.",
        "weakness_summary": f"Focus your next study session on: {', '.join(missing) or 'reviewing fundamentals'}.",
        "study_plan"      : [
            f"Re-read curriculum sections on: {', '.join(missing[:2]) or 'core concepts'}",
            "Attempt to explain each concept without looking at notes",
            "Come back and retake this assessment to track improvement",
        ],
        "motivation"      : "Every expert was once a beginner. Keep going! 🚀",
        "mastery_level"   : level,
        "avg_score"       : avg_score,
        "scores"          : scores,
    }


# ═══════════════════════════════════════════════════════════════════
# 5. Legacy explanation_engine (kept for backward compat)
# ═══════════════════════════════════════════════════════════════════
def explanation_engine(
    student_answer: str,
    misconception_report: Dict,
    retrieved_docs: List[str],
    graph_context: str = "",
) -> Dict:
    missing  = misconception_report.get("missing_concepts", [])
    score    = misconception_report.get("score", 0.0)
    graph_gap = misconception_report.get("graph_missing", [])

    if   score > 0.7: feedback = "Good understanding! You need only minor corrections."
    elif score > 0.4: feedback = "Partial understanding — a few key concepts are missing."
    else:             feedback = "Significant misunderstandings detected. Let's rebuild the concept."

    missing_str   = ", ".join(missing)  or "none identified"
    graph_str     = ", ".join(graph_gap) or "none"

    return {
        "feedback"  : feedback,
        "mistakes"  : f"Missing key terms: {missing_str}. Graph gaps: {graph_str}.",
        "correction": (retrieved_docs[0][:300] + "…") if retrieved_docs else "See your textbook.",
        "summary"   : f"Overlap: {int(score*100)}%. Focus on: {missing_str}.",
        "graph_ctx" : graph_context,
        "mode"      : "offline (rule-based)",
    }
