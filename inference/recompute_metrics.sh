#!/bin/bash

INFERENCE_DIR="/home/ivanfung/dm/inference_temp/result"

for INFERENCE_FILE in "${INFERENCE_DIR}"/*.json; do
    if [ -f "$INFERENCE_FILE" ]; then
        nohup python recalculate_all_metrics.py \
          --res_file_path "$INFERENCE_FILE" >> "recompute_metrics.log" 2>&1 &
        
        pid=$!
        wait $pid
    fi
done

