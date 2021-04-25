#!/bin/bash

timestamp() {
    date + "%T"
}

source venv/bin/activate;
python3 src/snp_500_dojo.py > logs/snp_bookcamp_logs_${timestamp}.txt;
deactivate;