#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.


# IMAGE_BSZ=96
# SUPCON_TEMPERATURE=0.07
# IMAGE_LEARNING_RATE=0.001
# IMAGE_LOSS_WEIGHT=0.02
# IMAGE_CONTRASTIVE_METHOD='SupCon'    #### or 'SimCLR'
# IMAGE_WARMUP_STEPS=0

## 48 0.07 1e-6 0.05 SupCon 0.0

IMAGE_BSZ=$1
SUPCON_TEMPERATURE=$2
IMAGE_LEARNING_RATE=$3
IMAGE_LOSS_WEIGHT=$4
IMAGE_CONTRASTIVE_METHOD=$5
IMAGE_WARMUP_STEPS=0
IMAGE_WEIGHT_DECAY=$6

OUTPUT_DIR=result/my-unsup-simcse-bert-base-uncased/$IMAGE_CONTRASTIVE_METHOD
OUTPUT_DIR=$OUTPUT_DIR-$IMAGE_BSZ
OUTPUT_DIR=$OUTPUT_DIR-$IMAGE_LOSS_WEIGHT
OUTPUT_DIR=$OUTPUT_DIR-$IMAGE_LEARNING_RATE
OUTPUT_DIR=$OUTPUT_DIR-$SUPCON_TEMPERATURE
OUTPUT_DIR=$OUTPUT_DIR-$IMAGE_WEIGHT_DECAY

python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --image_bsz $IMAGE_BSZ \
    --supcon_temperature $SUPCON_TEMPERATURE \
    --image_learning_rate $IMAGE_LEARNING_RATE \
    --image_loss_weight $IMAGE_LOSS_WEIGHT \
    --image_contrastive_method $IMAGE_CONTRASTIVE_METHOD \
    --image_warmup_steps $IMAGE_WARMUP_STEPS \
    --image_weight_decay $IMAGE_WEIGHT_DECAY \
    --new_supcon \
    --fp16


python evaluation.py \
    --model_name_or_path $OUTPUT_DIR \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test

echo $OUTPUT_DIR
