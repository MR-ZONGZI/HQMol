#!/bin/bash

GRAPH_TOWER="himol"
if [ "$GRAPH_TOWER" == "graphmvp" ]; then
    INIT_CHECKPOINT_GNN="../checkpoints/graphmvp.pth"
elif [ "$GRAPH_TOWER" == "moleculestm" ]; then
    INIT_CHECKPOINT_GNN="../checkpoints/MoleculeSTM/molecule_model.pth"
elif [ "$GRAPH_TOWER" == "himol" ]; then
    INIT_CHECKPOINT_GNN="../checkpoints/HiMol/himol.pth"
else
    echo "Not supported graph tower"
fi

HIMOL_FUSE="fuse"
PROJECTOR_TYPE="query" # options: [linear, query]
MODEL_PATH=../checkpoints/Graph-LLaVA/finetune_molcap/molcap-llava-himol-vicuna-v1-3-7b-finetune_lora-fuse-query
FINETUNE_CHECKPOINT_GNN=../checkpoints/Graph-LLaVA/pretrain/llava-himol-vicuna-v1-3-7b-pretrain-fuse-query/graph_tower.pth

EPOCH=50
OUT_FILE=eval_result/$GRAPH_TOWER-chebi20-molcap-lora-${EPOCH}ep-${MOL_VERSION}-$PROJECTOR_TYPE.jsonl
SCORE_FILE=score_result/$GRAPH_TOWER-chebi20-molcap-lora-${EPOCH}ep-${MOL_VERSION}-$PROJECTOR_TYPE.jsonl

python -m llava.eval.model_molcap \
    --model-path $MODEL_PATH \
    --in-file ../data/ChEBI-20_data/test.txt \
    --answers-file $OUT_FILE \
    --graph_tower $GRAPH_TOWER \
    --himol_fuse $HIMOL_FUSE \
    --num_head 4 \
    --gin_hidden_dim 512 \
    --graph-checkpoint-path $FINETUNE_CHECKPOINT_GNN \
    --model-base ../checkpoints/vicuna-v1-3-7b \
    --batch_size 1 \
    --temperature 0.2 \
    --add_selfies \
    --debug \

# evaluation 
python -m llava.eval.eval_molcap \
    --molcap_result_file $OUT_FILE \
    --text2mol_bert_path ../checkpoints/scibert_scivocab_uncased \
    --molcap_score_file $SCORE_FILE