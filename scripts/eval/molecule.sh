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


TASK=property_pred # retrosynthesis, reagent_pred, property_pred
EPOCH=5

LORA_MODEL_PATH=../checkpoints/Graph-LLaVA/pretrain/llava-himol-vicuna-v1-3-7b-pretrain-fuse-query
PATH_TO_PROPERTY_PREDICTION_TEST=../data/Molecule-oriented_Instructions/property_prediction_test.json
OUT_FILE=eval_result/$GRAPH_TOWER-$TASK-lora-fuse-${EPOCH}ep-query.jsonl
SCORE_FILE=score_result/$GRAPH_TOWER-$TASK-lora-fuse-${EPOCH}ep-query.jsonl
PRETRAIN_CHECKPOINT_GNN=../checkpoints/Graph-LLaVA/pretrain/llava-himol-vicuna-v1-3-7b-pretrain-fuse-query/graph_tower.pth

# Sampling
python -m llava.eval.molecule_metrics.generate_sample \
    --task $TASK \
    --model-path $LORA_MODEL_PATH \
    --graph_tower $GRAPH_TOWER \
    --himol_fuse fuse \
    --num_head 4 \
    --gin_hidden_dim 512 \
    --in-file $PATH_TO_PROPERTY_PREDICTION_TEST \
    --answers-file $OUT_FILE \
    --graph-checkpoint-path $PRETRAIN_CHECKPOINT_GNN \
    --model-base ../checkpoints/vicuna-v1-3-7b \
    --batch_size 1 --temperature 0.2 --top_p 1.0 \
    --add-selfies \
    --debug \

# Evaluation for reagent_pred or retrosynthesis
# python -m llava.eval.molecule_metrics.mol_translation_selfies \
#     --eval_result_file $OUT_FILE \
#     --score_result_file $SCORE_FILE \

# Evaluation for property_pred
# python -m llava.eval.molecule_metrics.property_metrics \
#     --eval_result_file $OUT_FILE \
#     --score_result_file $SCORE_FILE \