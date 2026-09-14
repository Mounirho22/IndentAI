# Amazon Bedrock AgentCore Deployment

This folder contains the deployment artifacts for packaging and running **IdentAI** with **Amazon Bedrock AgentCore** and AWS container runtimes.

## Architecture

```
Teacher Browser / API Client
           │ HTTP / JSON
           ▼
Amazon Bedrock AgentCore Runtime (Port 8080)
┌────────────────────────────────────────────────────────┐
│  agentcore/entrypoint.py                               │
│    └─ uvicorn main:app                                 │
│        ├─ FastAPI routes                               │
│        ├─ Strands Agent (strands-agents)               │
│        │   └─ BedrockModel (Strands runtime)          │
│        ├─ 10 Strands @tool Functions                   │
│        │   └─ boto3 Bedrock Converse (structured JSON) │
│        └─ SQLite (aiosqlite) lessons.db & state        │
└────────────────────────────────────────────────────────┘
           │ AWS IAM Credentials / Network
           ▼
    Amazon Bedrock (Claude 3.5 Sonnet / Haiku)
```

## Files
- `Dockerfile`: Multi-stage Python 3.12 container definition packaging the full FastAPI app and Strands runtime.
- `entrypoint.py`: Container entrypoint managing ASGI execution and configuration verification.
- `deploy.sh`: Builds Docker image, authenticates with AWS ECR, and pushes the container.
- `invoke.sh`: Smoke test script querying `/health`, `/api/bedrock-check`, and `POST /missions`.

## Deployment Steps
1. Configure AWS credentials with access to Amazon Bedrock and ECR:
   ```bash
   aws configure
   ```
2. Build and push to ECR:
   ```bash
   chmod +x agentcore/deploy.sh agentcore/invoke.sh
   ./agentcore/deploy.sh us-east-1 identai-agentcore latest
   ```
3. Test locally or against deployed runtime:
   ```bash
   ./agentcore/invoke.sh http://localhost:8080 "Nested loops"
   ```
