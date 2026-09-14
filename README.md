# IdentAI — Good Neighbor Agents 🤝

> **Autonomous Mentor Operations for Community Coding Centers — an AI agent that does the repetitive work silently in the background and only surfaces when a human judgment genuinely matters.**
>
> Built with the **Strands Agents SDK** on **Amazon Bedrock**, containerized for **Amazon Bedrock AgentCore**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![AWS Strands](https://img.shields.io/badge/AWS-Strands%20Agents-orange)](https://github.com/awslabs/strands-agents)
[![Amazon Bedrock](https://img.shields.io/badge/Amazon-Bedrock-232F3E)](https://aws.amazon.com/bedrock/)
[![Amazon AgentCore](https://img.shields.io/badge/Amazon-Bedrock%20AgentCore-FF9900)](https://aws.amazon.com/bedrock/agentcore/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-58%20passed-brightgreen)](#testing)

---

## The Pitch

**IdentAI is a Good Neighbor Agent for non-profit community coding centers: it quietly takes over the repetitive side of mentoring — checking every student submission, running their code in a sandbox, grading it against unit tests, tracking progress in a SQLite ledger — and it interrupts a volunteer mentor only when the situation is a genuine judgment call, handing over a one-screen Escalation Decision Brief and resuming the moment the mentor answers.** Under the hood it is a Strands Agents SDK runtime reasoning on Amazon Bedrock foundation models, deployed as an Amazon Bedrock AgentCore container, with a deterministic fallback that keeps every guarantee true even with zero model credentials.

## The Problem

Volunteer-run community coding centers run on goodwill and thin margins:

- **Mentor burnout is structural.** One volunteer mentor typically faces 10–25 learners across different tiers. Most of their evening is consumed by mechanical work — checking whether a loop prints the right numbers, re-explaining the same three syntax rules, updating progress sheets — not by the mentoring that volunteers actually signed up for.
- **Administrative friction loses students.** Progress lives in scattered spreadsheets (if it exists at all). A learner who is quietly stuck for two weeks becomes next week's dropout, because nobody had the bandwidth to notice the pattern.
- **The alternative — a chatbot — makes it worse.** A chat window puts every single interaction back on the human: someone has to judge every answer, copy every result, and remember every follow-up. Automation that adds judgment work is not automation.

## The Solution: An Agent That Earns Its Interruptions

IdentAI runs **autonomously in the background**. Student submissions (or any learning events) stream into its queue; the agent continuously evaluates each one — deterministic syntax analysis, sandboxed unit-test runs against the bounded pseudo-code interpreter, milestone progress evaluation — and **commits every routine outcome silently** to the progress ledger with a full telemetry trail. No notification, no dashboard ping, no mentor tap on the shoulder.

The agent surfaces for a human **exactly three** times, each with a structured **Escalation Decision Brief**:

| Trigger | Meaning | The human decides |
|---|---|---|
| `stuck_loop` | A student failed more than 3 consecutive attempts | approve (resume support) / reject (stop & redirect) / advise (inject a hint) |
| `hallucination_risk` | A claimed output contradicts deterministic execution | confirm the claim / confirm the mismatch / advise |
| `permission_gate` | A student earned an advanced module that stays locked | grant / withhold the unlock |

One call — `resolve_decision(decision_id, human_input)` — applies the mentor's verdict and the agent resumes background work immediately. **Everything else is zero-attention.**

The same philosophy holds at the infrastructure layer: the agent runs in an **Amazon Bedrock AgentCore** container as a background runtime, ingesting events and answering structured task invocations with an explicit `"AUTONOMOUS_COMPLETION"` or `"HUMAN_DECISION_REQUIRED"` verdict — never a chat message.

### But that's not all — it also designs the curriculum

The repo ships a second, equally autonomous Strands agent workflow: given a topic, it plans and generates tiered teaching materials (struggling / on-level / advanced), audits its own output through a three-layer quality engine (structural validation, the bounded code interpreter, a pedagogical misconception check), self-corrects within a bounded retry budget, and escalates pedagogical judgment calls to the teacher's review desk. See `demo_mission.py` and the review desk at `/`.

---

## Architecture at a Glance

```
 student submits ──▶  ingest  ──▶  SILENT PIPELINE  ──▶  committed to SQLite
 (web / API /          queue      syntax analysis        (ledger + telemetry)
  AgentCore invoke)               unit tests                     │
                                  milestone evaluation          ▼
                                       │                  quiet for humans
                                       │
                            ┌── routine? ── YES ──▶ auto-commit, keep going
                            └──── NO (3 gates only)
                                       │
                                       ▼
                        ⚡ ESCALATION DECISION BRIEF  ──▶  mentor
                                       ▲     (stuck_loop / hallucination_risk /
                                        │      permission_gate)
                                        └── resolve_decision(id, input)
                                                │
                                       agent resumes autonomously
```

> 📐 **Full architecture documentation** — Mermaid + ASCII diagrams, component responsibility matrix, and data-flow detail — lives in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

**Stack:**

| Layer | Technology |
|---|---|
| Agent orchestration | **Strands Agents SDK** (`strands.Agent`, `@tool` routing, `AgentState`, `FileSessionManager`, `InterventionHandler`) |
| Reasoning | **Amazon Bedrock** foundation models (Anthropic Claude via `BedrockModel`), LiteLLM/Gemini fallback, fully deterministic offline mode |
| Container runtime | **Amazon Bedrock AgentCore** (ECR + `bedrock-agentcore-control`, linux/amd64, non-root image) |
| Persistence | **SQLite** (`data/lessons.db`): submissions, milestones, decision briefs, telemetry, materials, curriculum missions |
| Service & UI | FastAPI + server-rendered mentor review desk |

---

## Project Layout

```
├── app/
│   ├── main.py                  # FastAPI service: review desk, mission orchestrator, learning-agent routes
│   └── interpreter/             # the agent runtime package
│       ├── engine.py            # bounded interpreter + AutonomousAgentEngine (background loop, HITL gate)
│       ├── evaluator.py         # deterministic silent assessment (syntax / unit tests / milestones)
│       └── models.py            # typed execution results
├── agentcore/                   # Amazon Bedrock AgentCore deployment
│   ├── Dockerfile               # hardened non-root container (linux/amd64)
│   ├── entrypoint.py            # dual mode: HTTP server / one-shot agent task JSON
│   ├── deploy.sh                # sanity checks -> ECR -> push -> AgentCore registration
│   └── invoke.sh                # smoke-test invocations
├── demos/
│   ├── demo_scenario.py         # ★ the live 3-phase cinematic demo (run this first)
│   └── demo_mission.py          # curriculum-mission lifecycle demo
├── scripts/
│   ├── check_bedrock.py         # five-point Bedrock reachability diagnostic
│   └── clean_repo.sh            # repo hygiene + MIT license compliance check
├── tests/                       # pytest suite (unit + integration), all green
├── data/lessons.db              # SQLite runtime data (gitignored, auto-created)
├── docs/ARCHITECTURE.md         # architecture deep-dive
├── verify.py                    # one-command installation health check
└── LICENSE                      # MIT
```

---

## Prerequisites

- **Python 3.10+** (3.12 recommended)
- **AWS account** with Amazon Bedrock model access — *or nothing at all*: the agent degrades to a fully deterministic mode when no credentials are present, so you can clone and demo offline
- **Docker** + **AWS CLI** (only for the AgentCore deployment path)

## Installation

```bash
# 1. Clone and enter the repository
git clone <your-public-repo-url>.git
cd identai

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 4. Configure the environment (optional for offline mode)
cp .env.example .env                 # if provided; otherwise create .env:
```

```dotenv
# .env — the only required setting is the provider
MODEL_PROVIDER=bedrock               # bedrock | litellm | mock (offline)
BEDROCK_REGION=us-east-1
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0

# AWS credentials — either environment variables…
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
# …or the standard ~/.aws/credentials / SSO / IAM role chain works.

# Optional fallback provider
# MODEL_PROVIDER=litellm
# GEMINI_API_KEY=your_gemini_api_key
```

```bash
# 5. Validate the whole installation with one command
python verify.py                     # environment + deps + database + pytest
python verify.py --deep              # adds the 230-check deep pipeline suite
```

## Run the Service

```bash
python main.py                       # http://127.0.0.1:8000 — mentor review desk + API
```

The autonomous learning agent starts with the app and begins draining its queue immediately.

**Learning-agent API surface** (the agent's only human-facing edges):

```bash
# Submit a learning event (fire-and-forget; the agent assesses in the background)
curl -X POST http://127.0.0.1:8000/student-submissions \
  -H "Content-Type: application/json" \
  -d '{"student_id":"stu-1","milestone_id":"while-loops",
       "pseudocode":"SET n TO 0\nWHILE n < 3\nSET n TO n + 1\nEND WHILE\nPRINT n",
       "expected_output":["3"]}'

# The mentor's work queue: open Escalation Decision Briefs
curl http://127.0.0.1:8000/agent/decisions

# Resolve a brief — the agent resumes autonomously
curl -X POST http://127.0.0.1:8000/agent/decisions/<decision_id>/resolve \
  -H "Content-Type: application/json" \
  -d '{"resolution":"advise","human_input":"Remember: the loop body must change the variable it tests."}'
```

## 🎬 The Live Demo (60 seconds, judge-friendly)

```bash
python demos/demo_scenario.py
```

Three cinematic phases in your terminal, all driven by the **real** engine and a real SQLite ledger:

1. **Silent Background Ingestion** — three students submit work; the agent evaluates code, runs unit tests, and updates the progress ledger with *zero human attention drained*.
2. **Cognitive Loop Detection & Escalation** — a fourth student keeps hitting the same conceptual wall; past the retry threshold the agent pauses and surfaces an **Educator Escalation Alert**.
3. **Human Decision & Autonomous Resolution** — you play the mentor: unlock simplified scaffolding, flag a 1-on-1 breakout, or auto-generate a tailored hint. The agent applies your directive, updates the records, and resumes.

Recording a video? Use `--auto` (hands-free) and `--speed 0.7` for pacing:

```bash
python demos/demo_scenario.py --auto
```

There is also a curriculum-design lifecycle demo: `python demos/demo_mission.py`.

## ☁️ Deploy to Amazon Bedrock AgentCore

One command builds, pushes, and registers the runtime:

```bash
export AGENTCORE_ROLE_ARN=arn:aws:iam::<account>:role/IdentAIAgentCoreRole
./agentcore/deploy.sh us-east-1 identai-agentcore latest --invoke
```

The script performs inline sanity checks (AWS credentials via STS, region validity, Docker daemon, Bedrock reachability), authenticates Docker to **ECR**, builds the hardened `linux/amd64` container, pushes it, registers/updates the **AgentCore runtime** (`aws bedrock-agentcore-control`), and can smoke-invoke it with a real agent task. The runtime name defaults to `identai-agent`; if the IAM role does not exist yet, the script prints the exact `aws iam` commands to create it.

The container is dual-mode: serve HTTP (`python agentcore/entrypoint.py --serve`) or answer one structured agent task per invocation:

```bash
echo '{"action":"assess_submission","student_id":"stu-9","milestone_id":"while-loops",
       "pseudocode":"WHILE n < 3\nPRINT n\nEND WHILE","expected_output":["3"]}' \
  | python agentcore/entrypoint.py
# → {"status": "AUTONOMOUS_COMPLETION", "outcome": "auto_passed", ...}
# → or {"status": "HUMAN_DECISION_REQUIRED", "decision": { ...brief... }}
```

## Testing

```bash
python -m pytest tests/ -v           # unit + integration, 58 tests
python verify.py                     # the full installation health check
```

The suite covers the bounded interpreter, the silent assessment layer, the autonomy boundary (routine outcomes never escalate; the three gates always do), HITL resolution semantics, SQLite state mutations, non-blocking background execution, provider fallback/mock modes, and the full curriculum-mission pipeline.

## License

MIT — see [LICENSE](LICENSE). The license is also declared in the repository **About** section.

---

<div align="center">
<sub>Built for the AWS "Agents for Humans" Hackathon — Good Neighbor Agents track.<br/>
Strands Agents SDK · Amazon Bedrock · Amazon Bedrock AgentCore · FastAPI · SQLite</sub>
</div>
