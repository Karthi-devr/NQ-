#!/usr/bin/env bash
# Build script for Render.com deployment
set -o errexit

pip install -r requirements.txt

cd trading_project
python manage.py collectstatic --no-input
