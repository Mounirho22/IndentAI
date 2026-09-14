#!/usr/bin/env bash
# agentcore/invoke.sh — Smoke-test invocation for IdentAI on AgentCore / container runtime.
#
# Usage:
#   ./agentcore/invoke.sh [ENDPOINT_URL] [TOPIC]
#
set -euo pipefail

ENDPOINT_URL="${1:-http://localhost:8080}"
TOPIC="${2:-Nested loops in pseudo-code}"

echo "=========================================================="
echo " Invoking IdentAI AgentCore runtime"
echo " Endpoint: $ENDPOINT_URL"
echo " Topic:    $TOPIC"
echo "=========================================================="

echo "1. Checking health endpoint..."
curl -s -f "${ENDPOINT_URL}/health" | jq . || curl -s "${ENDPOINT_URL}/health"
echo ""

echo "2. Running Bedrock connectivity diagnostic..."
curl -s "${ENDPOINT_URL}/api/bedrock-check" | jq . || curl -s "${ENDPOINT_URL}/api/bedrock-check"
echo ""

echo "3. Submitting curriculum mission..."
RESPONSE=$(curl -s -X POST "${ENDPOINT_URL}/missions" \
  -H "Content-Type: application/json" \
  -d "{\"topic\": \"$TOPIC\", \"unit_name\": \"Unit 3: Control Flow\", \"objective\": \"Students can trace and write nested loops in pseudo-code.\"}")

echo "Response:"
echo "$RESPONSE" | jq . || echo "$RESPONSE"
echo ""

MISSION_ID=$(echo "$RESPONSE" | jq -r '.mission_id // empty')
if [ -n "$MISSION_ID" ]; then
    echo "Tracking mission status: ${ENDPOINT_URL}/api/curriculum-missions/${MISSION_ID}..."
    curl -s "${ENDPOINT_URL}/api/curriculum-missions/${MISSION_ID}" | jq . || true
fi
