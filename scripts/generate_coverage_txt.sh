#!/bin/bash

# Run pytest with coverage
python -m pytest --cov=src | tee coverage.txt

# Append the generation message
echo " " >> coverage.txt
echo " " >> coverage.txt
echo -e "\n\nGenerated using './scripts/generate_coverage_txt.sh' in the terminal in the root folder of the project" >> coverage.txt
