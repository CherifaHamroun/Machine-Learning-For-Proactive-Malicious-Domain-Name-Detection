#!/bin/bash
pushd ./code_implementation/code_data/code_preprocessors
python3 data_completion_attributes_generation.py
python3 data_instances_generation.py
python3 data_preprocessing.py
python3 data_preprocessing_for_multiclass.py
popd


