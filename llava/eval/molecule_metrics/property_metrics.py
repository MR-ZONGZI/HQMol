import json
import argparse
from sklearn.metrics import mean_absolute_error
from typing import List

def compute_maes(eval_result_file:str):
    
    homo_gts, homo_preds = [], []  
    lumo_gts, lumo_preds = [], []  
    homo_lumo_gts, homo_lumo_preds = [], [] 

    with open(eval_result_file, 'r', encoding='utf-8') as file:
        data_array = json.load(file)
        for data in data_array:
            if "prompt" not in data or "gt_self" not in data or "pred_self" not in data:
                print("Warning: Missing necessary keys in the current line")
                continue

            prompt_value = data["prompt"]
            gt_value = float(data["gt_self"])
            pred_value = float(data["pred_self"])

            if "HOMO-LUMO" in prompt_value:
                homo_lumo_gts.append(gt_value)
                homo_lumo_preds.append(pred_value)
            elif "HOMO" in prompt_value and "LUMO" in prompt_value:
                homo_lumo_gts.append(gt_value)
                homo_lumo_preds.append(pred_value)
            elif "HOMO" in prompt_value:
                homo_gts.append(gt_value)
                homo_preds.append(pred_value)
            elif "LUMO" in prompt_value:
                lumo_gts.append(gt_value)
                lumo_preds.append(pred_value)
            else:
                print("Warning: Neither HOMO nor LUMO found in prompt")

    homo_mae = mean_absolute_error(homo_gts, homo_preds) if homo_gts and homo_preds else None
    lumo_mae = mean_absolute_error(lumo_gts, lumo_preds) if lumo_gts and lumo_preds else None
    homo_lumo_mae = mean_absolute_error(homo_lumo_gts, homo_lumo_preds) if homo_lumo_gts and homo_lumo_preds else None

    return homo_mae, lumo_mae, homo_lumo_mae


def compute_mae(eval_result_file:str, except_idxs:List[int]=[]):
    with open(eval_result_file) as f:
        results = json.load(f)
        gts = []
        preds = []
        for i, result in enumerate(results):
            if i in except_idxs:
                continue
            pred = result['pred_self']
            gt = result['gt_self']
            gts.append(float(gt))
            preds.append(float(pred))
        return mean_absolute_error(gts, preds)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_result_file", type=str, required=True)
    parser.add_argument("--score_result_file", type=str, required=True)
    args = parser.parse_args()
    # read except_idxs
    # with open('/cto_labs/AIDD/DATA/Mol-Instructions/Molecule-oriented_Instructions/property_overlap.txt', 'r') as f:
    #     except_idxs = [int(line.split('\t')[0]) for line in f.readlines()]
    homo_mae, lumo_mae, homo_lumo_mae = compute_maes(args.eval_result_file)
    avg_mae = compute_mae(args.eval_result_file)
    print(f"MAE of HOMO: {homo_mae}")
    print(f"MAE of LUMO: {lumo_mae}")
    print(f"MAE of HOMO-LUMO: {homo_lumo_mae}")
    print(f"Average MAE: {avg_mae}")

    with open(args.score_result_file, 'w') as json_file:
        json_file.write(json.dumps({
            "homo_mae": homo_mae,
            "lumo_mae": lumo_mae,
            "homo_lumo_mae": homo_lumo_mae,
            "avg_mae": avg_mae
        }, ensure_ascii=False) + '\n')

    print(f"结果已保存到 {args.score_result_file}")
    
    
"""
# property_pred
TASK=property_pred
EPOCH=5
GRAPH_TOWER=moleculestm
python -m llava.eval.molecule_metrics.property_metrics \
    --eval_result_file=eval_result/$GRAPH_TOWER-$TASK-${EPOCH}ep.jsonl
"""