#!/bin/bash
# Entrypoint for Render

# Export token
export DERIV_API_TOKEN=$(grep api_token config.yaml | awk '{print $2}')

# Run bot
python main.py