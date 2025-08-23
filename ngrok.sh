#!/bin/bash

APP_PORT=8001
NGROK_DOMAIN="jay-fit-safely.ngrok-free.app"

screen -dmS diagnosis bash -c "uv run python -m diagnosis_service"
screen -dmS ngrok bash -c "ngrok http --domain=${NGROK_DOMAIN} ${APP_PORT}"

echo "App running in 'diagnosis' screen session."
echo "ngrok running in 'ngrok' screen session."
echo "Use 'screen -ls' to list sessions and 'screen -r <name>' to attach."