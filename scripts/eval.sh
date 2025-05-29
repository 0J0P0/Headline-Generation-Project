#!/bin/bash

# run with bash!!! i.e. bash src/eval.sh -pre

echo "Starting headline generation evaluation..."

# add -pre if you want to evaluate the pre-trained BART model
PYTHONPATH=$(pwd) python src/evaluate.py "$@"

echo "Evaluation finished!"