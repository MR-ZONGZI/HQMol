#!/bin/bash

# Set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

GRAPH_TOWER="himol"
if [ "$GRAPH_TOWER" == "graphmvp" ]; then
    INIT_CHECKPOINT_GNN="./checkpoints/graphmvp.pth"
elif [ "$GRAPH_TOWER" == "moleculestm" ]; then
    INIT_CHECKPOINT_GNN="../checkpoints/MoleculeSTM/molecule_model.pth"
elif [ "$GRAPH_TOWER" == "himol" ]; then
    INIT_CHECKPOINT_GNN="../checkpoints/HiMol/himol.pth"
else
    echo "Not supported graph tower"
fi

TASK="molcap"
HIMOL_FUSE="fuse"
PROJECTOR_TYPE="query" # options: [linear, query]
DATA_PATH="path_to_your_pkl_data"
CHECKPOINT_FOLDER_PREFIX="../checkpoints/Graph-LLaVA"
PRETRAIN_PATH="$CHECKPOINT_FOLDER_PREFIX/pretrain/llava-$GRAPH_TOWER-$MODEL_VERSION-pretrain-$HIMOL_FUSE-$PROJECTOR_TYPE"
OUTPUT_DIR="$CHECKPOINT_FOLDER_PREFIX/finetune_molcap/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-finetune_lora-$HIMOL_FUSE-$PROJECTOR_TYPE"

deepspeed ../llava/train/train_mem.py \
    --deepspeed zero2.json \
    --lora_enable True \
    --model_name_or_path ../checkpoints/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path $DATA_PATH \
    --graph_tower $GRAPH_TOWER \
    --himol_fuse $HIMOL_FUSE \
    --num_head 4 \
    --mm_projector_type $PROJECTOR_TYPE \
    --tune_graph_tower True \
    --gin_hidden_dim 512 \
    --init_checkpoint $PRETRAIN_PATH/graph_tower.pth \
    --pretrain_mm_mlp_adapter $PRETRAIN_PATH/mm_projector.bin \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 50 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 8e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --report_to none
