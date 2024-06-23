#!/bin/bash
# custom config
DATA=data/
TRAINER=MultiModalAdapter

DATASET=$1
SEED=$2
CFG=$3
LOADER=$4
SHOTS=16

DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch ${LOADER} \
    --eval-only
fi