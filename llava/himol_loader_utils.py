import argparse
from typing import Tuple, Type

_MODULE_PATHS = {
    "moleculestm": "llava.model.llava_graph_arch",
    "himol": "llava.model.llava_higraph_arch",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_tower', type=str, required=False, default="himol", help="Graph tower type")
    parser.add_argument('--himol_fuse', type=str, required=False, default=None, help="Graph fusion type")
    args, unknown_args = parser.parse_known_args()
    return args

def import_graph_tower_module(graph_tower: str) -> Tuple[Type, Type]:
    if graph_tower not in _MODULE_PATHS:
        raise ValueError(f"Unsupported graph_tower: {graph_tower}")

    module_path = _MODULE_PATHS[graph_tower]
    try:
        module = __import__(module_path, fromlist=["LlavaMetaModel", "LlavaMetaForCausalLM"])
        return module.LlavaMetaModel, module.LlavaMetaForCausalLM
    except ImportError as e:
        raise ImportError(f"Failed to import module from {module_path}: {e}")

def get_args():
    return parse_args()

def get_llava_models():
    args = get_args()
    return import_graph_tower_module(args.graph_tower)

def get_graph_tower():
    args = get_args()
    return args.graph_tower

def get_himol_fuse():
    args = get_args()
    return args.himol_fuse

_args = None
_LlavaMetaModel, _LlavaMetaForCausalLM = None, None

def _init_globals():
    global _args, _LlavaMetaModel, _LlavaMetaForCausalLM, _HIMOL_FUSE
    if _args is None:
        _args = get_args()
        _LlavaMetaModel, _LlavaMetaForCausalLM = import_graph_tower_module(_args.graph_tower)
        _HIMOL_FUSE = _args.himol_fuse

_init_globals()

LlavaMetaModel = _LlavaMetaModel
LlavaMetaForCausalLM = _LlavaMetaForCausalLM