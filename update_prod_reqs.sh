#!/bin/bash

source ./venv/bin/activate
pip install pipreqs
pipreqs .

deactivate