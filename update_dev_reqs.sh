#!/bin/bash

source ./venv/bin/activate
pip check 
pip freeze > requirements_dev.txt

deactivate