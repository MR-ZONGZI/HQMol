from .lazy_supervised_dataset import LazySupervisedDataset, LazySupervisedGraphDataset
from .reagent_pred_dataset import ReagentPredSupervisedGraphDataset
from .retrosynthesis_dataset import RetrosynthesisSupervisedGraphDataset
from .property_pred_dataset import PropertyPredSupervisedGraphDataset
from .collators import DataCollatorForSupervisedDataset, GraphDataCollatorForSupervisedDataset
from torch.utils.data import ConcatDataset


def build_dataset(tokenizer, data_args):
    data_type = data_args.data_type
    if data_type == "supervised":
        dataset = LazySupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "reagent_pred":
        dataset = ReagentPredSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "retrosynthesis":
        dataset = RetrosynthesisSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    elif data_type == "property_pred":
        dataset = PropertyPredSupervisedGraphDataset(
            data_path=data_args.data_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
    else:
        raise NotImplementedError(f"Unknown data type: {data_type}")
    return dataset