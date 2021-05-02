#!/bin/bash

timestamp=$(date "+%Y_%m_%d-%H_%M_%S");
echo 'Running SNP Dojo: Start at ${timestamp}';
source venv/bin/activate;
python3 src/snp_500_dojo.py > logs/snp_bookcamp_logs_$timestamp.txt;
deactivate;
echo "End at $(date '+%Y_%m_%d-%H_%M_%S')";