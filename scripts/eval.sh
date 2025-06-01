#!/bin/bash

echo "Starting headline generation evaluation..."

PYTHONPATH=$(pwd) python src/evaluate.py "bart"

echo "Evaluation finished!"