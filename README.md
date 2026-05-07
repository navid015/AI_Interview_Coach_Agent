---
title: AI Interview Coach Agent
emoji: 🎤
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
license: mit
---

# 🎯 AI Interview Coach Agent

A voice-powered AI interview coach that conducts real technical interviews, assesses your answers in real-time, and generates personalized improvement reports.

## ✨ Features

- **🎤 Voice-Based Interaction** — Record your answer verbally; coach responds with voice
- **🤖 Adaptive AI Questioning** — Claude adjusts question difficulty based on your answers
- **📊 Real-Time Scoring** — 4-dimension assessment: Technical, Depth, Communication, Problem Solving
- **🏆 Personalized Report** — Detailed final report with 30-day improvement plan
- **💼 10 Role Types** — AI Engineer, Agentic AI Engineer, GenAI Engineer, LLM Engineer + more
- **🔊 Multiple Coach Voices** — Jenny, Guy, Sonia, Ryan (Edge TTS)

## 🚀 Setup

1. Get your **Anthropic API Key** from [console.anthropic.com](https://console.anthropic.com)
2. Enter it in the app's **API Key** field
3. Select your target role and interview settings
4. Click **Start Interview** and begin practicing!

> ⚠️ **Privacy note:** Your API key is held in this Gradio session's state and is sent directly to Anthropic from the server running this app. Do not enter your key on a public Space you do not trust. For private use, duplicate this Space or run locally.

## 🛠️ Tech Stack

- **LLM**: Anthropic Claude (claude-sonnet-4-5-20250929) with Tool Use API
- **STT**: OpenAI Whisper (base model, runs locally)
- **TTS**: Microsoft Edge TTS (edge-tts)
- **UI**: Gradio 4.x with custom CSS

## 📋 Interview Roles Supported

| Role | Focus Areas |
|------|------------|
| AI Engineer | ML, Deep Learning, PyTorch, MLOps |
| Agentic AI Engineer | Agent architectures, tool use, multi-agent systems |
| GenAI Engineer | LLMs, RAG, fine-tuning, vector DBs |
| LLM Engineer | Transformers, RLHF, inference optimization |
| ML Engineer | Classical ML, feature engineering, model serving |
| Data Scientist | Statistics, hypothesis testing, visualization |
| MLOps Engineer | CI/CD, model monitoring, cloud platforms |
| Software Engineer | DSA, system design, design patterns |
| Backend Engineer | APIs, databases, scalability |
| Full Stack Engineer | Frontend, backend, deployment |

## 🔧 Running Locally

```bash
# System dependency
sudo apt-get install ffmpeg   # macOS: brew install ffmpeg

# Python dependencies
pip install -r requirements.txt

# Run
python app.py
```

The app will be available at `http://localhost:7860`. Whisper's `base` model (~140 MB) downloads on first run.

## 🐛 Troubleshooting

- **"No assessment returned from Claude"** — usually a transient API issue; submit again.
- **Slow first response** — Whisper preloads at startup, but the first Claude call may be slower due to cold caches.
- **Empty audio / "couldn't hear your answer"** — record at least a few seconds of speech; very short or silent clips are rejected.
- **Rate limit errors** — wait a few seconds and resubmit. Free-tier API keys have lower limits.