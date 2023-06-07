import torch
from src.utils.classification import ToxicClassifier
import pickle
from argparse import ArgumentParser, Namespace

from loguru import logger
from torch.utils.data import Dataset

from src.data_sets.vtr_dataset import VTRDataset
from src.utils.common import load_json
from src.utils.config import TransformerConfig, TrainingConfig, VTRConfig
from src.utils.train_fine_tune import train


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--train-data", type=str, default=f"resources/data/train_dataset.jsonl", help="Path to train dataset."
    )
    arg_parser.add_argument("--val-data", type=str, default=None, help="Path to validation dataset.")
    arg_parser.add_argument("--test-data", type=str, default=None, help="Path to test dataset.")
    arg_parser.add_argument(
        "--char2array",
        type=str,
        default="resources/char2array.pkl",
        help="Path to char2array [only for VTR model].",
    )
    arg_parser.add_argument("--no-ocr", action="store_true", help="Do not use OCR with visual models.")

    arg_parser = VTRConfig.add_to_arg_parser(arg_parser)
    arg_parser = TransformerConfig.add_to_arg_parser(arg_parser)
    arg_parser = TrainingConfig.add_to_arg_parser(arg_parser)
    return arg_parser


def train_vtr_encoder(args: Namespace, train_data: list, val_data: list = None, test_data: list = None):
    model_config = TransformerConfig.from_arguments(args)
    training_config = TrainingConfig.from_arguments(args)
    vtr = VTRConfig.from_arguments(args)

    with open(args.char2array, "rb") as f:
        char2array = pickle.load(f)

    dataset_args = (char2array, vtr.window_size, vtr.stride, training_config.max_seq_len)
    train_dataset: Dataset = VTRDataset(train_data, *dataset_args)
    val_dataset: Dataset = VTRDataset(val_data, *dataset_args) if val_data else None
    test_dataset: Dataset = VTRDataset(test_data, *dataset_args) if test_data else None

    path = "wandb/run-20230602_193133-27p8gai1/files/pretrained.pt"
    pretrained = torch.load(path)
    model = ToxicClassifier(model_config.emb_size, 2, pretrained)

    train(
        model,
        train_dataset,
        training_config,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )


def main(args: Namespace):

    logger.info("Loading data...")
    train_data = load_json(args.train_data)
    val_data = load_json(args.val_data) if args.val_data else None
    test_data = load_json(args.test_data) if args.test_data else None

    train_vtr_encoder(args, train_data, val_data, test_data)


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(_args)
