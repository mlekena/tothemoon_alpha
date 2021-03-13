#!/bin/bash

source ./venv/bin/activate
pip install -r requirements_dev.txt
pip check 

deactivate