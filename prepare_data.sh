#!/bin/bash
mkdir -p data/quora
python quora_data_process.py
python generate_data.py data
cp data/test_input.txt data/dev_input.txt
cp data/test_ref.txt data/dev_ref.txt
