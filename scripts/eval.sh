#!/bin/bash

echo "Starting headline generation evaluation..."

PYTHONPATH=$(pwd) python src/evaluate.py

echo "Evaluation finished!"