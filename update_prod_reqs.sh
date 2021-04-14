#!/bin/bash

source ./venv/bin/activate
pip install pipreqs --force
pipreqs .

deactivate