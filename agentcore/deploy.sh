#!/usr/bin/env bash
# agentcore/deploy.sh — build & deploy IdentAI to Amazon Bedrock AgentCore.
#
# Pipeline: sanity checks -> ECR auth -> docker build (linux/amd64) -> push
#           -> register/update the AgentCore runtime -> optional smoke invoke.
#
# Usage:
#   ./agentcore/deploy.sh [AWS_REGION] [ECR_REPO_NAME] [IMAGE_TAG] [--invoke]
#
# Environment:
#   AGENTCORE_ROLE_ARN   IAM role the runtime assumes (required to register).
#                        Create one with:  ./agentcore/deploy.sh --create-role
#   AGENT_RUNTIME_NAME   Runtime name (default: identai-agent)
#
# Examples:
#   ./agentcore/deploy.sh us-east-1
#   AGENTCORE_ROLE_ARN=arn:aws:iam::123456789012:role/agentcore-role \
#     ./agentcore/deploy.sh us-east-1 identai-agentcore v1 --invoke
#
set -euo pipefail

# ---------------------------------------------------------------- arguments --
AWS_REGION="${1:-${AWS_REGION:-us-east-1}}"
ECR_REPO_NAME="${2:-${ECR_REPO_NAME:-identai-agentcore}}"
IMAGE_TAG="${3:-${IMAGE_TAG:-latest}}"
INVOKE_SMOKE="${4:-}"
AGENT_RUNTIME_NAME="${AGENT_RUNTIME_NAME:-identai-agent}"

BLUE='\033[0;34m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
step() { printf "\n${BLUE}==>${NC} %s\n" "$1"; }
ok()   { printf "    ${GREEN}✓${NC} %s\n" "$1"; }
warn() { printf "    ${YELLOW}!${NC} %s\n" "$1"; }
die()  { printf "\n${RED}ERROR:${NC} %s\n" "$1" >&2; exit 1; }

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=========================================================="
echo " IdentAI -> Amazon Bedrock AgentCore"
echo " Region: $AWS_REGION | Repo: $ECR_REPO_NAME | Tag: $IMAGE_TAG"
echo "=========================================================="

# ------------------------------------------------------- 0. sanity checks --
step "0/6 Sanity checks (tools, credentials, region)"
command -v aws    >/dev/null 2>&1 || die "AWS CLI is not installed. https://docs.aws.amazon.com/cli/"
command -v docker >/dev/null 2>&1 || die "Docker is not installed. https://docs.docker.com/get-docker/"
docker info >/dev/null 2>&1      || die "Docker daemon is not running (start Docker Desktop / dockerd)."
[ -f "agentcore/Dockerfile" ]    || die "Run this script from the repository root (agentcore/Dockerfile not found)."
ok "aws CLI + docker daemon present"

# Credentials: a real STS call, not an env-var guess — expired sessions and
# missing profiles both surface here with an actionable message.
IDENTITY="$(aws sts get-caller-identity --output json 2>/dev/null)" \
  || die "AWS credentials did not resolve. Run 'aws configure' or 'aws sso login' and retry."
ACCOUNT_ID="$(printf '%s' "$IDENTITY" | grep -o '"Account": *"[^"]*"' | head -1 | sed 's/.*"Account": *"//;s/"//')"
ok "authenticated as account $ACCOUNT_ID"

# Region: must be a real region name, not e.g. 'us-east-1a'.
case "$AWS_REGION" in
  us-|eu-|ap-|ca-|sa-|af-|me-|il-) : ;;
  *) die "'$AWS_REGION' does not look like an AWS region (expected e.g. us-east-1)." ;;
esac
aws ec2 describe-regions --region "$AWS_REGION" --query 'Regions[?RegionName==`'"$AWS_REGION"'`].RegionName' \
  --output text 2>/dev/null | grep -q "$AWS_REGION" \
  || die "Region $AWS_REGION is not reachable with these credentials."
ok "region $AWS_REGION valid and reachable"

# Bedrock access check (non-fatal: the container degrades to deterministic
# mode, but a judge demo wants real model calls).
if aws bedrock list-foundation-models --region "$AWS_REGION" >/dev/null 2>&1; then
  ok "Bedrock control plane reachable in $AWS_REGION"
else
  warn "Bedrock list-foundation-models failed — model invocation may be unavailable (check model access)."
fi

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"

# ------------------------------------------------------------ 1. ECR repo --
step "1/6 Ensuring ECR repository exists"
aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$AWS_REGION" >/dev/null 2>&1 \
  || aws ecr create-repository \
       --repository-name "$ECR_REPO_NAME" --region "$AWS_REGION" \
       --image-scanning-configuration scanOnPush=true \
       --query 'repository.repositoryUri' --output text
ok "repository ready: $ECR_REPO_NAME"

# --------------------------------------------------------- 2. ECR auth -----
step "2/6 Authenticating Docker to ECR"
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com" \
  >/dev/null
ok "ECR login token stored"

# --------------------------------------------------------- 3. build --------
step "3/6 Building image (linux/amd64 — AgentCore requirement)"
docker build --platform linux/amd64 -t "$ECR_URI" -f agentcore/Dockerfile .
ok "built $ECR_URI"

# --------------------------------------------------------- 4. push --------
step "4/6 Pushing image to ECR"
docker push "$ECR_URI"
ok "pushed"

# -------------------------------------------- 5. AgentCore registration ---
step "5/6 Registering the AgentCore runtime"
if [ -z "${AGENTCORE_ROLE_ARN:-}" ]; then
  warn "AGENTCORE_ROLE_ARN is not set — skipping runtime registration."
  cat <<ROLE

    The container is on ECR. To register the AgentCore runtime you need an IAM
    role the service can assume. Create one, then re-run with it exported:

      aws iam create-role --role-name IdentAIAgentCoreRole \\
        --assume-role-policy-document '{
          "Version": "2012-10-17",
          "Statement": [{"Effect": "Allow", "Principal": {"Service":
            "bedrock-agentcore.amazonaws.com"}, "Action": "sts:AssumeRole"}]
        }'
      aws iam put-role-policy --role-name IdentAIAgentCoreRole \\
        --policy-name IdentAIAgentCoreInline \\
        --policy-document '{
          "Version": "2012-10-17",
          "Statement": [
            {"Effect": "Allow", "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"], "Resource": "*"},
            {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"], "Resource": "*"},
            {"Effect": "Allow", "Action": ["ecr:GetAuthorizationToken", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"], "Resource": "*"}
          ]}'
      export AGENTCORE_ROLE_ARN=arn:aws:iam::${ACCOUNT_ID}:role/IdentAIAgentCoreRole
      ./agentcore/deploy.sh ${AWS_REGION} ${ECR_REPO_NAME} ${IMAGE_TAG}

ROLE
else
  RUNTIME_ID="$(aws bedrock-agentcore-control list-agent-runtimes --region "$AWS_REGION" \
      --query "agentRuntimes[?agentRuntimeName=='${AGENT_RUNTIME_NAME}'].agentRuntimeId | [0]" \
      --output text 2>/dev/null || true)"
  if [ -n "$RUNTIME_ID" ] && [ "$RUNTIME_ID" != "None" ]; then
    aws bedrock-agentcore-control update-agent-runtime \
      --agent-runtime-id "$RUNTIME_ID" \
      --agent-runtime-artifact "{\"containerConfiguration\": {\"containerUri\": \"${ECR_URI}\"}}" \
      --role-arn "$AGENTCORE_ROLE_ARN" \
      --network-configuration '{"networkMode": "PUBLIC"}' \
      --protocol-configuration '{"serverProtocol": "HTTP"}' \
      --environment-variables "{\"MODEL_PROVIDER\": \"bedrock\", \"BEDROCK_REGION\": \"${AWS_REGION}\"}" \
      --region "$AWS_REGION" \
      --query 'agentRuntimeArn' --output text
    ok "updated existing runtime $AGENT_RUNTIME_NAME ($RUNTIME_ID)"
  else
    RUNTIME_ARN="$(aws bedrock-agentcore-control create-agent-runtime \
      --agent-runtime-name "$AGENT_RUNTIME_NAME" \
      --agent-runtime-artifact "{\"containerConfiguration\": {\"containerUri\": \"${ECR_URI}\"}}" \
      --role-arn "$AGENTCORE_ROLE_ARN" \
      --network-configuration '{"networkMode": "PUBLIC"}' \
      --protocol-configuration '{"serverProtocol": "HTTP"}' \
      --environment-variables "{\"MODEL_PROVIDER\": \"bedrock\", \"BEDROCK_REGION\": \"${AWS_REGION}\"}" \
      --description "IdentAI autonomous learning agent (Strands + Bedrock)" \
      --region "$AWS_REGION" \
      --query 'agentRuntimeArn' --output text)"
    ok "created runtime: $RUNTIME_ARN"
  fi
fi

# --------------------------------------------------------- 6. smoke invoke -
step "6/6 Optional smoke invoke"
if [ "${INVOKE_SMOKE}" == "--invoke" ]; then
  RUNTIME_ARN="${RUNTIME_ARN:-}"
  [ -n "$RUNTIME_ARN" ] || RUNTIME_ARN="$(aws bedrock-agentcore-control list-agent-runtimes \
      --region "$AWS_REGION" \
      --query "agentRuntimes[?agentRuntimeName=='${AGENT_RUNTIME_NAME}'].agentRuntimeArn | [0]" \
      --output text 2>/dev/null || true)"
  [ -n "$RUNTIME_ARN" ] && [ "$RUNTIME_ARN" != "None" ] \
    || die "No runtime ARN to invoke (register the runtime first)."
  PAYLOAD='{"action":"assess_submission","student_id":"smoke-1","milestone_id":"loops","pseudocode":"SET counter TO 0\nREPEAT 2 TIMES\nSET counter TO counter + 1\nEND REPEAT\nPRINT counter","expected_output":["2"]}'
  echo "$PAYLOAD" > /tmp/agentcore-smoke.json
  aws bedrock-agentcore invoke-agent-runtime \
    --agent-runtime-arn "$RUNTIME_ARN" \
    --content-type "application/json" \
    --payload "file:///tmp/agentcore-smoke.json" \
    --region "$AWS_REGION" \
    --query 'response' --output text || warn "smoke invoke failed — check the runtime logs in CloudWatch."
  rm -f /tmp/agentcore-smoke.json
else
  printf "    ${YELLOW}•${NC} %s\n" "skip (pass --invoke as the 4th argument to smoke-test the runtime)"
fi

echo ""
echo "=========================================================="
echo " Deployment complete."
echo "   Image:   $ECR_URI"
echo "   Runtime: $AGENT_RUNTIME_NAME (region $AWS_REGION)"
echo "=========================================================="
