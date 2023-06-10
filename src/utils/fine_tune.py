import torch
import json

from src.utils.classification import ToxicClassifier
import pickle
from argparse import ArgumentParser, Namespace

from loguru import logger
from torch.utils.data import Dataset
from src.data_sets.common import (
    AugmentationDataset,
    SlicesDataset,
    SlicesIterableDataset,
    TokenizedDataset,
    TokenizedIterableDataset,
    SlicesIterableDatasetOCR,
    SlicesDatasetOCR,
)
from src.data_sets.vtr_dataset import VTRDataset
from src.utils.common import load_json
from src.utils.config import TransformerConfig, TrainingConfig, VTRConfig, AugmentationConfig
from src.utils.train_fine_tune import train
from src.data_sets.translation_datasets import ToxicDataset
from src.utils.augmentation import init_augmentations
from src.data_sets.vtr_dataset import VTRDataset, VTRDatasetOCR
from src.models.vtr.ocr import OCRHead


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
    arg_parser.add_argument("--pretrained-path", type=str, help="Path to pre-trained.")

    arg_parser = VTRConfig.add_to_arg_parser(arg_parser)
    arg_parser = TransformerConfig.add_to_arg_parser(arg_parser)
    arg_parser = TrainingConfig.add_to_arg_parser(arg_parser)
    arg_parser = AugmentationConfig.add_to_arg_parser(arg_parser)
    return arg_parser


def train_vtr_encoder(
    args: Namespace, train_data: list, val_data: list = None, test_data: list = None, pretrained=None
):
    model_config = TransformerConfig.from_arguments(args)
    training_config = TrainingConfig.from_arguments(args)
    augmentation_config = AugmentationConfig.from_arguments(args)
    vtr = VTRConfig.from_arguments(args)

    with open(args.char2array, "rb") as f:
        char2array = pickle.load(f)

    with open(augmentation_config.leet, "r", encoding="utf-8") as json_file:
        leet_symbols = json.load(json_file)

    with open(augmentation_config.clusters, "rb") as f:
        cluster_symbols = pickle.load(f)

    augmentations = init_augmentations(
        expected_changes_per_word=augmentation_config.expected_changes_per_word,
        cluster_symbols=cluster_symbols,
        leet_symbols=leet_symbols,
    )

    train_dataset = ToxicDataset(train_data)
    train_dataset = AugmentationDataset(
        dataset=train_dataset,
        augmentations=augmentations,
        proba_per_text=augmentation_config.proba_per_text,
        expected_changes_per_text=augmentation_config.expected_changes_per_text,
        max_augmentations=augmentation_config.max_augmentations,
    )

    dataset_args = (char2array, vtr.window_size, vtr.stride, training_config.max_seq_len)
    if args.no_ocr:
        train_dataset = SlicesIterableDataset(train_dataset, char2array)
        val_dataset: Dataset = VTRDataset(val_data, *dataset_args) if val_data else None
        test_dataset: Dataset = VTRDataset(test_data, *dataset_args) if test_data else None

        model = ToxicClassifier(
            2,
            pretrained,
            emb_size=model_config.emb_size,
            n_head=model_config.n_head,
            n_layers=model_config.num_layers,
            height=vtr.font_size,
            width=vtr.window_size,
        )

    else:
        train_dataset = SlicesIterableDatasetOCR(train_dataset, char2array)
        val_dataset = VTRDatasetOCR(val_data, ratio=vtr.ratio, *dataset_args) if val_data else None
        test_dataset = VTRDatasetOCR(test_data, ratio=vtr.ratio, *dataset_args) if test_data else None

        # charset = val_dataset.char_set | test_dataset.char_set
        # char2int_dict = {char: i + 1 for i, char in enumerate(charset)}
        char2int_dict = {char: i + 1 for i, char in enumerate(char2array.keys())}
        logger.info(
            f"OCR parameters: hidden size: {vtr.hidden_size_ocr}, # layers: {vtr.num_layers_ocr}, "
            f"# classes: {len(char2array.keys())}"
        )

        ocr = OCRHead(
            input_size=vtr.font_size,
            hidden_size=vtr.hidden_size_ocr,
            num_layers=vtr.num_layers_ocr,
            num_classes=len(char2array.keys()),
            # num_classes=len(charset),
        )

        model = ToxicClassifier(
            2,
            pretrained,
            emb_size=model_config.emb_size,
            n_head=model_config.n_head,
            n_layers=model_config.num_layers,
            height=vtr.font_size,
            width=vtr.window_size,
            ocr=ocr,
            char2int=char2int_dict,
            alpha=vtr.alpha,
        )

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
    pretrained = torch.load(args.pretrained_path) if args.pretrained_path else None

    train_vtr_encoder(args, train_data, val_data, test_data, pretrained)


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(_args)
