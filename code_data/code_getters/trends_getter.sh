#!/bin/bash
mkdir -p ./code_implementation/code_data/data_trends
pushd ./code_implementation/code_data/code_getters
python3 twitter_trends_getter.py
python3 google_trends_getter.py
popd
