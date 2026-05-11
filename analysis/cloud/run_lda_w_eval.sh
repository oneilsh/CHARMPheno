# Pick a unique run id; the SAVE_DIR is where eval will look later.
RUN_ID=lda_k50_$(date +%Y%m%d_%H%M%S)
SAVE_DIR=/mnt/gcs/$BUCKET/runs/$RUN_ID

make lda-bq-smoke \
    SAVE_DIR=$SAVE_DIR \
    SAVE_INTERVAL=10 \
    LDA_BQ_ARGS='--K 50 --max-iter 150 --person-mod 50 --vocab-size 5000 --min-df 10 --holdout-fraction 0.1 --holdout-seed 42 --tau0 64 --kappa 0.7 --print-topics-every 1'


make eval-bq-coherence CHECKPOINT=$SAVE_DIR