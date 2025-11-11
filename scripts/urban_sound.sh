#!/bin/bash
DATASET="UrbanSound8K"
METHOD=$1
ATTACK=$2

N_CLASSES=10 # Number of classes in the Dataset

if [ "$METHOD" != "zeroshot" ] && [ "$METHOD" != "coop" ] && [ "$METHOD" != "cocoop" ] && [ "$METHOD" != "palm" ]; then
    echo "Invalid METHOD=$METHOD . Please choose one of the following: ['zeroshot', 'coop', 'cocoop', 'palm']"
    exit 1
fi

if [ "$ATTACK" != "trojanwave" ] && [ "$ATTACK" != "flowmur" ] && [ "$ATTACK" != "nbad" ] && [ "$ATTACK" != "nba" ] && [ "$ATTACK" != "noattack" ]; then
    echo "Invalid ATTACK=$ATTACK . Please choose one of the following: ['trojanwave', 'flowmur', 'nbad', 'nba', 'noattack']"
    exit 1
fi

echo "Running METHOD=$METHOD on DATASET=$DATASET with ATTACK=$ATTACK"

DATASET_ROOT="$DATA/Audio-Datasets/$DATASET"

if [ -d "$DATASET_ROOT" ]; then
    echo "Dataset path exists: $DATASET_ROOT"
else
    echo "Dataset path does not exist. Please set the correct path to the dataset root directory in variable DATASET_ROOT"
fi

if [ "$METHOD" = "coop" ] || [ "$METHOD" = "cocoop" ]; then
    CTX_DIM=512
else
    CTX_DIM=1024
fi


SEED=0
TARGET_LABELS=0
# TARGET_LABELS=$(seq 0 $((N_CLASSES-1)))

for TARGET_LABEL in $TARGET_LABELS
    do
        echo "Running with SEED=$SEED and TARGET_LABEL=$TARGET_LABEL"

        if [ -f "$DATASET_ROOT/train.csv" ]; then rm -rf "$DATASET_ROOT/train.csv"; fi
        if [ -f "$DATASET_ROOT/test.csv" ]; then rm -rf "$DATASET_ROOT/test.csv"; fi
        cp "$DATASET_ROOT/csv_files/train.csv" "$DATASET_ROOT/train.csv"
        cp "$DATASET_ROOT/csv_files/test.csv" "$DATASET_ROOT/test.csv"

        python main.py \
            --method_name $METHOD \
            --dataset_root $DATASET_ROOT \
            --n_epochs 25 \
            --test_model_last_epoch_only \
            --ctx_dim $CTX_DIM \
            --batch_size 16 \
            --lr 0.05 \
            --seed $SEED \
            --exp_name $DATASET \
            --num_shots 8 \
            --attack_name $ATTACK \
            --poison_rate 0.0 \
            --target_label $TARGET_LABEL \
            --eps 0.2 \
            --rho 0.1 \
            --lambda_clean 2.0 \
            --lambda_adv 1.0 \
            --do_logging \
            --load_model_path $DATA/models/trojanwave-models-attack \
            --save_model \
            --save_model_path $DATA/models/trojanwave-models-defense  
    done