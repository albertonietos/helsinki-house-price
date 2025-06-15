#!/bin/bash
# Manual linting script for flake8
echo "Running flake8 linting..."
flake8 --ignore=E501,E203,W503 --max-line-length=88 app/ src/ helsinkihouse/
echo "Linting complete!"
