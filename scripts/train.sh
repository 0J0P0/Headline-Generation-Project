#!/bin/bash

echo "Starting headline generation training..."

PYTHONPATH=$(pwd) python src/train.py "bart"

echo "Training finished!"
