#!/usr/bin/env bash

make hdp-bq-smoke HDP_BQ_ARGS='\
  --T 100 \
  --K 20 \
  --save-dir ./hdp_run \
  --save-interval 1 \
  --max-iter 4 \
  --person-mod 500 \
  --subsampling-rate 0.2 \
  --print-topics-every 1'