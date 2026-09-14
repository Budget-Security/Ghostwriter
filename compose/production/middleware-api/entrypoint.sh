#!/bin/sh
set -e

CERT_DIR="/ssl"
CERT_FILE="$CERT_DIR/ghostwriter.crt"
KEY_FILE="$CERT_DIR/ghostwriter.key"

# DOCX generation + LibreOffice PDF conversion run far past gunicorn's default
# 30s worker timeout, which kills the worker mid-request and surfaces as a 500.
exec gunicorn \
  --bind 0.0.0.0:8443 \
  --certfile "$CERT_FILE" \
  --keyfile "$KEY_FILE" \
  --workers "${GUNICORN_WORKERS:-2}" \
  --timeout "${GUNICORN_TIMEOUT:-900}" \
  --graceful-timeout "${GUNICORN_GRACEFUL_TIMEOUT:-60}" \
  --access-logfile - \
  --error-logfile - \
  app:app
