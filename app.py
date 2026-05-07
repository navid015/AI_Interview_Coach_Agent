"""
🎯 AI Interview Coach Agent
Voice-powered interview preparation with real-time assessment & personalized coaching
Built with Gradio + Anthropic Claude + Whisper + Edge-TTS
"""

import gradio as gr
import anthropic
import whisper
import asyncio
import edge_tts
import tempfile
import os
import json
import re
import time
import glob
from pathlib import Path

# ─────────────────────────────────────────────
# WORKAROUND: gradio_client 1.3.0 schema parser bug
# https://github.com/gradio-app/gradio/issues/10662
# The parser crashes on schemas containing `additionalProperties: True` (a bool)
# or other boolean-valued schema fields, because get_type() does `"const" in schema`
# which fails when schema is a bool. This patch makes the parser tolerate bools.
# ─────────────────────────────────────────────
try:
    import gradio_client.utils as _gc_utils

    _orig_get_type = _gc_utils.get_type

    def _safe_get_type(schema):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_get_type(schema)

    _gc_utils.get_type = _safe_get_type

    _orig_json_schema_to_python_type = _gc_utils._json_schema_to_python_type

    def _safe_json_schema_to_python_type(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_json_schema_to_python_type(schema, defs)

    _gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type
except Exception as _e:
    print(f"Warning: could not patch gradio_client schema parser: {_e}")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
# Updated to current Sonnet 4.5 (Sept 2025) — far better at structured tool use
# than the original May 2024 Sonnet 4. See https://docs.claude.com for latest.
MODEL = "claude-sonnet-4-5-20250929"

ROLES = [
    "AI Engineer", "Agentic AI Engineer", "GenAI Engineer", "LLM Engineer",
    "Machine Learning Engineer", "Data Scientist", "MLOps Engineer",
    "Software Engineer", "Backend Engineer", "Full Stack Engineer"
]

INTERVIEW_TYPES = ["Technical", "Behavioral", "System Design", "Mixed"]
LEVELS = ["Junior (0-2 yrs)", "Mid-Level (2-5 yrs)", "Senior (5-8 yrs)", "Staff/Principal (8+ yrs)"]
VOICES = {
    "Jenny (Female, US)": "en-US-JennyNeural",
    "Guy (Male, US)": "en-US-GuyNeural",
    "Sonia (Female, UK)": "en-GB-SoniaNeural",
    "Ryan (Male, UK)": "en-GB-RyanNeural"
}

ROLE_TOPICS = {
    "AI Engineer": "ML fundamentals, deep learning (CNNs/RNNs/Transformers), PyTorch/TensorFlow, model deployment, MLOps, production ML systems, A/B testing, feature engineering",
    "Agentic AI Engineer": "LLM agent architectures (ReAct, CoT, Plan-Execute), tool use/function calling, multi-agent systems (AutoGen/CrewAI/LangGraph), memory systems (episodic/semantic), agent evaluation, safety, LangChain/LlamaIndex",
    "GenAI Engineer": "LLMs (GPT/Claude/Gemini/Llama), prompt engineering & optimization, RAG systems, fine-tuning (LoRA/QLoRA/PEFT), vector databases (Pinecone/Weaviate/Chroma), embeddings, multimodal AI, LLM evaluation",
    "LLM Engineer": "Transformer architecture (attention, positional encoding), tokenization strategies, pre-training/fine-tuning paradigms, RLHF/RLAIF, inference optimization (quantization/distillation/vLLM), context window management, alignment techniques",
    "Machine Learning Engineer": "Classical ML algorithms (ensemble methods, SVMs), feature engineering, cross-validation, model monitoring/drift detection, experiment tracking (MLflow/W&B), model serving (TorchServe/BentoML), data pipelines",
    "Data Scientist": "Statistical inference, hypothesis testing, Bayesian methods, causal inference, data visualization, SQL optimization, business metric design, storytelling, AB testing design",
    "MLOps Engineer": "CI/CD for ML (GitHub Actions/Jenkins), model registry (MLflow/Neptune), monitoring/observability (Evidently/WhyLabs), Kubernetes/Docker, cloud ML platforms (SageMaker/Vertex/AzureML), DVC, feature stores (Feast)",
    "Software Engineer": "Data structures & algorithms, system design (CAP theorem, consistency), design patterns, API design (REST/GraphQL), testing strategies, distributed systems, code review practices",
    "Backend Engineer": "REST/GraphQL API design, relational & NoSQL databases, caching (Redis/Memcached), message queues (Kafka/RabbitMQ), microservices architecture, scalability patterns, security best practices",
    "Full Stack Engineer": "React/Vue/Angular (state management, hooks), Node.js/FastAPI backends, SQL & NoSQL databases, CI/CD pipelines, cloud deployment (AWS/GCP), responsive design, authentication/authorization"
}

# TTS chunking — Edge TTS handles long text fine, but very long inputs slow rendering.
TTS_CHUNK_LIMIT = 1500
TTS_TEMP_DIR = tempfile.gettempdir()

# ─────────────────────────────────────────────
# CSS STYLING
# ─────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

.gradio-container {
    background: linear-gradient(135deg, #0a0a1a 0%, #1a1040 50%, #0d1b2a 100%) !important;
    min-height: 100vh;
}

#app-header {
    text-align: center;
    padding: 36px 20px 28px;
    background: linear-gradient(135deg, rgba(102,126,234,0.12), rgba(118,75,162,0.12));
    border: 1px solid rgba(102,126,234,0.25);
    border-radius: 20px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

#app-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(102,126,234,0.08) 0%, transparent 70%);
    pointer-events: none;
}

#app-header h1 {
    font-size: 2.6em;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}

#app-header p {
    color: rgba(200,200,255,0.65);
    font-size: 1.05em;
    margin: 0;
}

.panel {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 18px !important;
    padding: 20px !important;
}

.score-box {
    background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(118,75,162,0.1)) !important;
    border: 1px solid rgba(102,126,234,0.25) !important;
    border-radius: 12px !important;
    padding: 14px !important;
}

.agent-log {
    background: rgba(0,10,30,0.7) !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
    border-radius: 12px !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.82em !important;
    color: #a5b4fc !important;
    min-height: 120px !important;
}

button.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1.05em !important;
    padding: 12px 20px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(102,126,234,0.3) !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(102,126,234,0.45) !important;
}

button.secondary {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    color: white !important;
    padding: 12px 20px !important;
    box-shadow: 0 4px 15px rgba(16,185,129,0.25) !important;
}

label { color: rgba(200,210,255,0.9) !important; font-weight: 500 !important; }

.gradio-dropdown select, .gradio-textbox textarea, .gradio-textbox input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: white !important;
}

.status-bar {
    background: rgba(16,185,129,0.1) !important;
    border: 1px solid rgba(16,185,129,0.3) !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    color: #6ee7b7 !important;
    font-weight: 500 !important;
}
"""

# ─────────────────────────────────────────────
# WHISPER + TTS HELPERS
# ─────────────────────────────────────────────
_whisper_model = None

def get_whisper():
    """Load Whisper lazily but allow preload at startup."""
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model (base)...")
        _whisper_model = whisper.load_model("base")
        print("Whisper loaded!")
    return _whisper_model


# Sentinel used to detect transcription failures unambiguously (not substring "error")
_TRANSCRIPT_ERR_PREFIX = "[TRANSCRIPTION_ERROR]"


def transcribe_audio(audio_path):
    if not audio_path:
        return ""
    try:
        model = get_whisper()
        result = model.transcribe(audio_path, fp16=False)
        return result["text"].strip()
    except Exception as e:
        return f"{_TRANSCRIPT_ERR_PREFIX} {e}"


def _cleanup_old_tts_files(max_age_seconds=600):
    """Best-effort cleanup of old edge-tts mp3 temp files to prevent /tmp filling up."""
    try:
        now = time.time()
        for path in glob.glob(os.path.join(TTS_TEMP_DIR, "tmp*.mp3")):
            try:
                if now - os.path.getmtime(path) > max_age_seconds:
                    os.remove(path)
            except OSError:
                pass
    except Exception:
        pass


async def _tts_async(text, voice):
    communicate = edge_tts.Communicate(text, voice)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        path = f.name
    await communicate.save(path)
    return path


def _run_async(coro):
    """
    Safely run an async coroutine even when an event loop already exists
    (which happens inside Gradio's request handling).
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Fall through to creating a fresh loop
            raise RuntimeError("loop already running")
        return loop.run_until_complete(coro)
    except RuntimeError:
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()


def text_to_speech(text, voice="en-US-JennyNeural"):
    """
    Convert text to speech via edge-tts. Handles long text by chunking on sentence
    boundaries instead of silently truncating to 600 chars (the original bug).
    For very long text, only the first chunk is returned (Gradio Audio takes one file)
    but the chunk is whole-sentence, not mid-word.
    """
    try:
        _cleanup_old_tts_files()

        # Strip markdown noise
        clean = re.sub(r'[#*`\[\]_]', '', text)
        clean = re.sub(r'\n+', '. ', clean).strip()

        if not clean:
            return None

        # If short enough, send as-is
        if len(clean) <= TTS_CHUNK_LIMIT:
            return _run_async(_tts_async(clean, voice))

        # Otherwise truncate on a sentence boundary so audio doesn't end mid-word
        truncated = clean[:TTS_CHUNK_LIMIT]
        last_period = max(truncated.rfind('. '), truncated.rfind('! '), truncated.rfind('? '))
        if last_period > TTS_CHUNK_LIMIT * 0.5:
            truncated = truncated[:last_period + 1]
        return _run_async(_tts_async(truncated, voice))
    except Exception as e:
        print(f"TTS error: {e}")
        return None


# ─────────────────────────────────────────────
# ANTHROPIC TOOL SCHEMA
# ─────────────────────────────────────────────
ASSESS_TOOL = {
    "name": "assess_answer",
    "description": "Assess the candidate's interview answer with structured scoring",
    "input_schema": {
        "type": "object",
        "properties": {
            "feedback_text": {
                "type": "string",
                "description": "Detailed, specific feedback on the answer (3-4 sentences)"
            },
            "technical_score": {"type": "integer", "minimum": 1, "maximum": 10},
            "depth_score": {"type": "integer", "minimum": 1, "maximum": 10},
            "communication_score": {"type": "integer", "minimum": 1, "maximum": 10},
            "problem_solving_score": {"type": "integer", "minimum": 1, "maximum": 10},
            "strengths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-3 specific strengths demonstrated"
            },
            "improvements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-3 specific areas to improve with actionable suggestions"
            },
            "next_question": {
                "type": "string",
                "description": "Next interview question to ask the candidate"
            }
        },
        "required": ["feedback_text", "technical_score", "depth_score", "communication_score",
                     "problem_solving_score", "strengths", "improvements", "next_question"]
    }
}


def get_system_prompt(role, itype, level):
    topics = ROLE_TOPICS.get(role, "core software engineering principles")
    return f"""You are a senior technical interviewer at a top-tier tech company (FAANG-level), specializing in {role} roles.

You are conducting a {itype} interview for a {level} candidate applying for a {role} position.

Key topics for {role}:
{topics}

Your interviewing principles:
- Ask ONE focused, well-crafted question at a time
- Questions must be appropriate for {level} — calibrate complexity accordingly
- For Technical: dive into specific implementations, trade-offs, and real scenarios
- For Behavioral: probe for STAR method (Situation, Task, Action, Result)
- For System Design: focus on scalability, trade-offs, real-world constraints
- Adapt subsequent questions based on what the candidate reveals about their knowledge
- Be fair but rigorous — this simulates a real FAANG interview
- Do NOT repeat questions you have already asked in this session

ALWAYS use the assess_answer tool to provide structured, actionable feedback.
Your feedback should be specific to their actual answer, not generic."""


# ─────────────────────────────────────────────
# CORE INTERVIEW LOGIC
# ─────────────────────────────────────────────
def _build_assessment_messages(state, transcript, extra_instruction=""):
    """
    Build a clean message list for the assessment call.

    Instead of accumulating full conversation history with malformed assistant
    turns (the original bug), we send a compact, well-formed context:
      - A single user message that contains all prior Q&A summaries plus the
        current question and the candidate's answer.

    This keeps the context grounded, prevents repeated questions, and avoids
    the "tool_use without tool_result" protocol violation.
    """
    qa_log = state.get("qa_log", [])  # list of {q, a, summary}

    parts = []
    if qa_log:
        parts.append("## Prior questions and answers in this session:\n")
        for i, qa in enumerate(qa_log, 1):
            parts.append(f"**Q{i}:** {qa['q']}")
            parts.append(f"**A{i} (transcribed):** {qa['a']}")
            if qa.get("summary"):
                parts.append(f"**Prior assessment summary:** {qa['summary']}")
            parts.append("")

    current_q = state.get("current_q", "")
    parts.append(f"## Current question (Q{state['q_num']}/{state['max_q']}):")
    parts.append(current_q)
    parts.append("")
    parts.append("## Candidate's answer (transcribed from voice):")
    parts.append(transcript)

    if extra_instruction:
        parts.append("")
        parts.append(extra_instruction)

    parts.append("")
    parts.append(
        "Use the assess_answer tool to score this answer and provide a "
        "DIFFERENT next question that builds on what they've shown. "
        "Do not repeat any prior question."
    )

    return [{"role": "user", "content": "\n".join(parts)}]


def start_interview(api_key, role, itype, level, voice_name, state):
    if not api_key or not api_key.strip():
        return (
            state,
            [{"role": "assistant", "content": "⚠️ **Please enter your Anthropic API key** in the setup panel to start your interview."}],
            None,
            "*Waiting for API key...*",
            "No scores yet",
            "Waiting for API key..."
        )

    voice = VOICES.get(voice_name, "en-US-JennyNeural")
    state = {
        "api_key": api_key.strip(),
        "role": role, "itype": itype, "level": level,
        "voice": voice,
        "qa_log": [],          # list of {q, a, summary}
        "current_q": "",       # the question currently being answered
        "q_num": 1, "max_q": 5,
        "scores": [], "all_strengths": [], "all_improvements": [],
        "started": True
    }

    steps = ["🚀 Initializing interview session...", f"  Role: {role} | Type: {itype} | Level: {level}"]

    try:
        client = anthropic.Anthropic(api_key=state["api_key"])
        steps.append("🧠 Generating first question with Claude...")

        resp = client.messages.create(
            model=MODEL,
            max_tokens=400,
            system=get_system_prompt(role, itype, level),
            messages=[{
                "role": "user",
                "content": f"Start the {itype} interview. Ask an excellent, specific opening question for a {level} {role}. Ask ONLY the question — no preamble, no greeting."
            }]
        )
        first_q = resp.content[0].text.strip()
        state["current_q"] = first_q

        steps.append("🔊 Converting question to speech...")
        greeting = f"Welcome to your {role} interview. I'll ask you 5 questions. Let's begin. Question 1: {first_q}"
        audio = text_to_speech(greeting, voice)

        steps.append("✅ Interview session started successfully!")

        chat = [
            {"role": "assistant", "content": f"👋 Welcome to your **{role}** — {itype} Interview!"},
            {"role": "assistant", "content": f"📋 **5 questions** tailored for a **{level}** candidate. Speak naturally — I'll assess your answer in real time."},
            {"role": "assistant", "content": f"---\n**❓ Question 1 / 5:**\n\n{first_q}"},
        ]

        return (
            state, chat, audio,
            f"✅ Live | **{role}** {itype} Interview | Q1/5",
            "🎙️ Record your answer, then click **Submit**",
            "\n".join(steps)
        )
    except anthropic.AuthenticationError:
        return (
            state,
            [{"role": "assistant", "content": "❌ **Invalid API key.** Please check your Anthropic API key and try again.\n\nYou can get one at https://console.anthropic.com"}],
            None,
            "Invalid API key",
            "No scores",
            "Authentication failed — invalid API key"
        )
    except Exception as e:
        return (
            state,
            [{"role": "assistant", "content": f"❌ **Error starting interview:** {str(e)}\n\nCheck your API key and network connection."}],
            None,
            "Error starting interview",
            "No scores",
            f"Error: {e}"
        )


def process_answer(audio_path, state, chat_history):
    if not state or not state.get("started"):
        return state, chat_history, None, "Start interview first", "No scores", "Interview not started"

    if audio_path is None:
        return state, chat_history, None, "⚠️ No audio recorded", _format_scores(state.get("scores", [])), "No audio detected — please record your answer"

    steps = []
    steps.append("🎧 [1/5] Transcribing audio with Whisper (base model)...")

    transcript = transcribe_audio(audio_path)

    # Fixed: check for the explicit sentinel, not the substring "error"
    # (an answer like "I'd handle the error by..." used to wrongly fail this check)
    if transcript.startswith(_TRANSCRIPT_ERR_PREFIX):
        steps.append(f"❌ Transcription failed: {transcript}")
        chat_history.append({"role": "assistant", "content": "❌ Could not transcribe audio. Please try again."})
        return state, chat_history, None, "Transcription failed", _format_scores(state.get("scores", [])), "\n".join(steps)

    # Guard against empty/silent recordings — would otherwise send empty content to Claude
    if not transcript or len(transcript.strip()) < 3:
        steps.append("⚠️ Transcript was empty or too short. Please record a real answer.")
        chat_history.append({"role": "assistant", "content": "⚠️ I couldn't hear your answer. Please record a longer response."})
        return state, chat_history, None, "Empty answer", _format_scores(state.get("scores", [])), "\n".join(steps)

    steps.append(f"✅ [2/5] Transcribed: \"{transcript[:80]}{'...' if len(transcript) > 80 else ''}\"")
    steps.append(f"🧠 [3/5] Analyzing response for {state['role']} competencies...")

    chat_history.append({"role": "user", "content": f"🎤 **You:** {transcript}"})

    client = anthropic.Anthropic(api_key=state["api_key"])
    is_last = state["q_num"] >= state["max_q"]

    try:
        steps.append("📊 [4/5] Scoring across 4 competency dimensions...")

        extra_instruction = (
            "This is the FINAL question of the interview, so the next_question field "
            "may contain a brief closing remark instead of a real question."
            if is_last else ""
        )

        messages = _build_assessment_messages(state, transcript, extra_instruction)

        tool_resp = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=get_system_prompt(state["role"], state["itype"], state["level"]),
            tools=[ASSESS_TOOL],
            tool_choice={"type": "tool", "name": "assess_answer"},
            messages=messages
        )

        # Extract tool_use input
        assessment = None
        for block in tool_resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "assess_answer":
                assessment = block.input
                break

        if not assessment:
            raise ValueError("No assessment returned from Claude — tool_use block missing")

        steps.append("✨ [5/5] Generating personalized feedback...")

        avg = (assessment["technical_score"] + assessment["depth_score"]
               + assessment["communication_score"] + assessment["problem_solving_score"]) / 4

        state["scores"].append({
            "technical": assessment["technical_score"],
            "depth": assessment["depth_score"],
            "communication": assessment["communication_score"],
            "problem_solving": assessment["problem_solving_score"],
            "avg": avg
        })
        state["all_strengths"].extend(assessment.get("strengths", []))
        state["all_improvements"].extend(assessment.get("improvements", []))

        # Append a clean Q&A record (with a short summary) to the log
        # — this is what gets fed into the NEXT turn's context, properly formatted.
        summary = (
            f"Score {avg:.1f}/10. Strengths: {'; '.join(assessment.get('strengths', []))}. "
            f"Gaps: {'; '.join(assessment.get('improvements', []))}."
        )
        state["qa_log"].append({
            "q": state["current_q"],
            "a": transcript,
            "summary": summary
        })

        # Format chat feedback
        strengths_md = "\n".join([f"  ✅ {s}" for s in assessment.get("strengths", [])])
        improv_md = "\n".join([f"  📈 {s}" for s in assessment.get("improvements", [])])
        score_bar = _bar(avg)
        grade = _grade(avg)

        feedback_md = f"""**📊 Q{state['q_num']} Assessment — Score: {avg:.1f}/10 [{score_bar}] {grade}**

{assessment['feedback_text']}

**What you did well:**
{strengths_md}

**Areas to strengthen:**
{improv_md}"""

        chat_history.append({"role": "assistant", "content": feedback_md})

        if not is_last:
            state["q_num"] += 1
            next_q = assessment.get("next_question", "Tell me about a challenging project you worked on.")
            state["current_q"] = next_q
            chat_history.append({"role": "assistant", "content": f"---\n**❓ Question {state['q_num']} / 5:**\n\n{next_q}"})

            tts_text = f"Score {avg:.0f} out of 10. {assessment['feedback_text'][:220]} Next question: {next_q}"
            audio = text_to_speech(tts_text, state["voice"])

            scores_md = _format_scores(state["scores"])
            status = f"✅ Q{state['q_num']-1} assessed | Score: {avg:.1f}/10 | Moving to Q{state['q_num']}"
            steps.append(f"🎙️ Ready for Q{state['q_num']}")

            return state, chat_history, audio, status, scores_md, "\n".join(steps)
        else:
            return _finish_interview(state, chat_history, assessment, steps)

    except anthropic.AuthenticationError:
        steps.append("❌ Authentication error — invalid API key")
        chat_history.append({"role": "assistant", "content": "❌ Invalid API key. Please restart the interview with a valid key."})
        return state, chat_history, None, "Auth error", _format_scores(state.get("scores", [])), "\n".join(steps)
    except anthropic.RateLimitError:
        steps.append("❌ Rate limit hit — please wait a moment and try again")
        chat_history.append({"role": "assistant", "content": "⏳ Rate limit reached. Please wait a moment, then submit again."})
        return state, chat_history, None, "Rate limited", _format_scores(state.get("scores", [])), "\n".join(steps)
    except Exception as e:
        steps.append(f"❌ Error: {str(e)}")
        chat_history.append({"role": "assistant", "content": f"❌ Processing error: {str(e)}"})
        return state, chat_history, None, f"Error: {e}", _format_scores(state.get("scores", [])), "\n".join(steps)


def _finish_interview(state, chat_history, last_assessment, steps):
    steps.append("📝 Generating comprehensive final report...")
    client = anthropic.Anthropic(api_key=state["api_key"])
    scores = state["scores"]

    overall = sum(s["avg"] for s in scores) / len(scores) if scores else 7.0
    tech_avg = sum(s["technical"] for s in scores) / len(scores) if scores else 7.0
    depth_avg = sum(s["depth"] for s in scores) / len(scores) if scores else 7.0
    comm_avg = sum(s["communication"] for s in scores) / len(scores) if scores else 7.0
    ps_avg = sum(s["problem_solving"] for s in scores) / len(scores) if scores else 7.0

    top_strengths = list(dict.fromkeys(state["all_strengths"]))[:5]
    top_improvements = list(dict.fromkeys(state["all_improvements"]))[:6]

    try:
        report_resp = client.messages.create(
            model=MODEL, max_tokens=2000,
            messages=[{"role": "user", "content": f"""Write a comprehensive, professional interview performance report for:

**Candidate Profile:**
- Role: {state['role']} | Level: {state['level']} | Interview Type: {state['itype']}
- Overall Score: {overall:.1f}/10 | Technical: {tech_avg:.1f} | Depth: {depth_avg:.1f} | Communication: {comm_avg:.1f} | Problem Solving: {ps_avg:.1f}
- Observed Strengths: {', '.join(top_strengths) if top_strengths else 'N/A'}
- Improvement Areas: {', '.join(top_improvements) if top_improvements else 'N/A'}

Write in Markdown with these sections:
## 🎯 Executive Summary
## 📊 Performance by Dimension
## 💪 Top 3 Strengths (with specific examples from interview)
## 🚀 Top 3 Critical Improvement Areas (with concrete action steps)
## 📚 Recommended Study Resources (specific books, courses, topics)
## ✅ Hiring Recommendation (Strong Yes / Yes / Maybe / No — with clear reasoning)
## 🗓️ 30-Day Improvement Plan (week by week)

Be honest, specific, and genuinely actionable. Avoid generic advice."""}]
        )
        report = report_resp.content[0].text
    except Exception as e:
        report = f"Report generation error: {e}"

    bar = _bar(overall)
    grade = _grade(overall)

    final_msg = f"""# 🏆 Interview Complete — Final Performance Report

## Overall Score: **{overall:.1f}/10** `[{bar}]` — **{grade}**

| Dimension | Score | Visual |
|-----------|-------|--------|
| 🔧 Technical Accuracy | **{tech_avg:.1f}**/10 | `{_bar(tech_avg)}` |
| 🔬 Depth of Knowledge | **{depth_avg:.1f}**/10 | `{_bar(depth_avg)}` |
| 💬 Communication | **{comm_avg:.1f}**/10 | `{_bar(comm_avg)}` |
| 🧩 Problem Solving | **{ps_avg:.1f}**/10 | `{_bar(ps_avg)}` |

---

{report}"""

    chat_history.append({"role": "assistant", "content": final_msg})

    tts_text = f"Interview complete! Overall score: {overall:.1f} out of 10, grade {grade}. {report[:400]}"
    audio = text_to_speech(tts_text, state.get("voice", "en-US-JennyNeural"))

    steps.append("🏆 Final report complete!")
    scores_md = _format_scores(state["scores"], final=True, overall=overall)

    return (
        state, chat_history, audio,
        f"🏁 **Interview Complete!** Final Score: {overall:.1f}/10 | Grade: {grade}",
        scores_md,
        "\n".join(steps)
    )


def _grade(score):
    """Full A-F grading scale (original only went down to C)."""
    if score >= 9: return "🏅 A+"
    if score >= 8: return "🥇 A"
    if score >= 7.5: return "⭐ B+"
    if score >= 7: return "✅ B"
    if score >= 6: return "📈 C+"
    if score >= 5: return "📝 C"
    if score >= 4: return "⚠️ D"
    return "❌ F"


def _bar(value, total=10):
    """Render a 10-char bar. Uses round() so 7.9 → 8 bars instead of truncating to 7."""
    filled = max(0, min(total, round(value)))
    return "█" * filled + "░" * (total - filled)


def _format_scores(scores_list, final=False, overall=None):
    if not scores_list:
        return "*Scores will appear after each answer*"
    lines = []
    if final and overall is not None:
        lines.append(f"### 🏆 Final: **{overall:.1f}/10** `[{_bar(overall)}]`\n")
    for i, s in enumerate(scores_list, 1):
        avg = s.get("avg", 7)
        lines.append(f"**Q{i}:** {avg:.1f}/10 `{_bar(avg)}`")
    if len(scores_list) > 1:
        run_avg = sum(s.get("avg", 7) for s in scores_list) / len(scores_list)
        lines.append(f"\n📊 **Running Avg:** {run_avg:.1f}/10")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────
with gr.Blocks(title="🎯 AI Interview Coach") as demo:

    gr.HTML("""
    <div id="app-header">
        <h1>🎯 AI Interview Coach Agent</h1>
        <p>Voice-powered interview prep • Real-time AI assessment • Personalized coaching for tech roles</p>
    </div>
    """)

    state = gr.State({})

    with gr.Row(equal_height=False):
        # ── LEFT PANEL ──────────────────────────────
        with gr.Column(scale=1, elem_classes="panel"):
            gr.Markdown("### ⚙️ Interview Setup")

            api_key = gr.Textbox(
                label="🔑 Anthropic API Key",
                type="password",
                placeholder="sk-ant-api03-..."
            )
            role = gr.Dropdown(
                choices=ROLES, value="AI Engineer",
                label="💼 Target Role"
            )
            itype = gr.Dropdown(
                choices=INTERVIEW_TYPES, value="Mixed",
                label="📋 Interview Type"
            )
            level = gr.Dropdown(
                choices=LEVELS, value="Mid-Level (2-5 yrs)",
                label="📊 Experience Level"
            )
            voice_pick = gr.Dropdown(
                choices=list(VOICES.keys()), value="Jenny (Female, US)",
                label="🔊 Coach Voice"
            )
            start_btn = gr.Button("🚀 Start Interview", variant="primary")

            gr.Markdown("---")
            gr.Markdown("### 📊 Live Score Tracker")
            scores_out = gr.Markdown(
                "*Scores appear after each answer*",
                elem_classes="score-box"
            )

            gr.Markdown("---")
            gr.Markdown("""
### 📖 How to Use
1. 🔑 Enter your Anthropic API key
2. 🎯 Select your target role & interview type
3. 🚀 Click **Start Interview**
4. 🔊 Listen to the coach's question
5. 🎤 Click mic icon to record your answer
6. ✅ Click **Submit Answer**
7. 📊 Get instant AI feedback
8. 🔁 Repeat for all 5 questions
9. 🏆 Receive your **Final Report**

---
💡 **Tip:** Speak clearly and take your time. The AI coach adapts questions based on your answers!
            """)

        # ── RIGHT PANEL ─────────────────────────────
        with gr.Column(scale=2):
            status_md = gr.Markdown(
                "*Configure your interview settings on the left and click Start*",
                elem_classes="status-bar"
            )

            chatbot = gr.Chatbot(
                label="💬 Interview Session",
                height=400,
                show_label=True,
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=coach")
            )

            with gr.Row():
                coach_audio = gr.Audio(
                    label="🔊 Coach's Voice (Auto-plays)",
                    autoplay=True,
                    interactive=False
                )

            gr.Markdown("### 🎤 Your Answer")
            mic_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Record your answer — click the mic to start/stop",
                interactive=False
            )

            submit_btn = gr.Button(
                "✅ Submit Answer",
                variant="secondary",
                interactive=False
            )

            gr.Markdown("### 🤖 Agent Processing Trace")
            agent_log = gr.Textbox(
                label="Live processing steps",
                lines=5,
                interactive=False,
                elem_classes="agent-log",
                placeholder="Agent steps will appear here in real-time..."
            )

    # ── EVENT HANDLERS ───────────────────────────
    start_btn.click(
        fn=start_interview,
        inputs=[api_key, role, itype, level, voice_pick, state],
        outputs=[state, chatbot, coach_audio, status_md, scores_out, agent_log]
    ).then(
        fn=lambda: (gr.update(interactive=True), gr.update(interactive=True)),
        outputs=[mic_input, submit_btn]
    )

    submit_btn.click(
        fn=process_answer,
        inputs=[mic_input, state, chatbot],
        outputs=[state, chatbot, coach_audio, status_md, scores_out, agent_log]
    ).then(
        fn=lambda: gr.update(value=None),
        outputs=[mic_input]
    )


if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())