import os
import torch

from transformers import Trainer
from typing import Optional
from llava.himol_loader_utils import get_graph_tower, get_himol_fuse


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def get_keys_to_match(graph_tower, himol_fuse):
    if graph_tower == "moleculestm":
        return ['mm_projector']
    elif graph_tower == "himol":
        return ['mm_projector']
    else:
        raise ValueError(f"Unsupported GRAPH_TOWER value: {graph_tower}")


class LLaVATrainer(Trainer):

    def _save_checkpoint(self, model, trial, metrics=None):

        super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Adapters
            keys_to_match = get_keys_to_match(get_graph_tower(), get_himol_fuse())

            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = {}
            for key in keys_to_match:
                weight_to_save[key] = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), [key])

            # Graph Tower
            graph_weights = {}
            print("Trying to find the parameters of graph tower...")
            graph_weights = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), ['graph_tower'])
            # if getattr(self.args, 'tune_graph_tower', False):
            #     print("Trying to find the parameters of graph tower...")
            #     graph_weights = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), ['graph_tower'])
            # else:
            #     print("No need to find the parameters of graph tower...")

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)

                print("Saving Adapters...")
                for key, weight in weight_to_save.items():
                    torch.save(weight, os.path.join(output_dir, f'{key}.bin'))

                if graph_weights:  # save when weights are not none
                    print("Saving Graph Tower...")
                    torch.save(graph_weights, os.path.join(output_dir, 'graph_tower.pth'))
                else:
                    print("Can not find Graph Tower weights...")


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
