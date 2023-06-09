import json
import pickle
from argparse import ArgumentParser, Namespace

from loguru import logger
from transformers import NllbTokenizer

from src.data_sets.common import (
    AugmentationDataset,
    SlicesDataset,
    SlicesIterableDataset,
    TokenizedDataset,
    TokenizedIterableDataset,
    SlicesIterableDatasetOCR,
    SlicesDatasetOCR,
)
from src.data_sets.translation_datasets import FloresDataset, NLLBDatasetRuEn, ToxicDataset
from torch.utils.data import IterableDataset, Dataset
from src.models.embedders.ttr import TTREmbedder
from src.models.embedders.vtr import VTREmbedder
from src.models.tasks import SequenceClassifier
from src.utils.augmentation import init_augmentations
from src.utils.config import TransformerConfig, TrainingConfig, VTRConfig, AugmentationConfig
from src.utils.train_fine_tune import train
from src.utils.common import load_json
from src.models.vtr.ocr import OCRHead
from src.data_sets.vtr_dataset import VTRDataset, VTRDatasetOCR


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

    arg_parser.add_argument(
        "--probas",
        type=str,
        default="resources/nllb/probas_nllb.pkl",
        help="Path to probabilities of language pairs [for lang detect task].",
    )

    arg_parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Augmentations are not applied to texts.",
    )

    #arg_parser.add_argument("--dataset-dir", type=str, help="Directory of validation and test Flores files.")

    arg_parser.add_argument(
        "--tokenizer", type=str, default="resources/tokenizer", help="Path to tokenizer [only for vanilla model]."
    )

    arg_parser.add_argument("--vtr", action="store_true", help="Use Visual Token Representations.")
    arg_parser.add_argument("--no-ocr", action="store_true", help="Do not use OCR with visual models.")

    arg_parser = VTRConfig.add_to_arg_parser(arg_parser)
    arg_parser = TransformerConfig.add_to_arg_parser(arg_parser)
    arg_parser = TrainingConfig.add_to_arg_parser(arg_parser)
    arg_parser = AugmentationConfig.add_to_arg_parser(arg_parser)
    return arg_parser


def train_vtr_encoder(args: Namespace, train_data: list, val_data: list = None, test_data: list = None):
    model_config = TransformerConfig.from_arguments(args)
    training_config = TrainingConfig.from_arguments(args)
    augmentation_config = AugmentationConfig.from_arguments(args)

    with open(args.probas, "rb") as f:
        probas = pickle.load(f)

    with open(args.char2array, "rb") as f:
        char2array = pickle.load(f)

    with open(augmentation_config.leet, "r", encoding="utf-8") as json_file:
        leet_symbols = json.load(json_file)

    with open(augmentation_config.clusters, "rb") as f:
        cluster_symbols = pickle.load(f)

    vtr = VTRConfig.from_arguments(args)
    channels = (1, 64, 128, vtr.out_channels)

    augmentations = init_augmentations(
        expected_changes_per_word=augmentation_config.expected_changes_per_word,
        cluster_symbols=cluster_symbols,
        leet_symbols=leet_symbols,
    )

    train_dataset: IterableDataset | Dataset

    embedder: TTREmbedder | VTREmbedder
    train_dataset = ToxicDataset(train_data)
    train_dataset = AugmentationDataset(
        dataset=train_dataset,
        augmentations=augmentations,
        proba_per_text=augmentation_config.proba_per_text,
        expected_changes_per_text=augmentation_config.expected_changes_per_text,
        max_augmentations=augmentation_config.max_augmentations,
    )

    if args.vtr:

        embedder = VTREmbedder(
            height=vtr.font_size,
            width=vtr.window_size,
            conv_kernel_size=vtr.conv_kernel_size,
            pool_kernel_size=vtr.pool_kernel_size,
            emb_size=model_config.emb_size,
            channels=channels,
        )

        dataset_args = (char2array, vtr.window_size, vtr.stride, training_config.max_seq_len)
        if args.no_ocr:
            train_dataset = SlicesIterableDataset(train_dataset, char2array)
            val_dataset: Dataset = VTRDataset(val_data, *dataset_args) if val_data else None
            test_dataset: Dataset = VTRDataset(test_data, *dataset_args) if test_data else None

            model = SequenceClassifier(model_config, embedder, training_config.max_seq_len)

        else:
            train_dataset = SlicesIterableDatasetOCR(train_dataset, char2array)
            val_dataset = VTRDatasetOCR(val_data, ratio=vtr.ratio, *dataset_args) if val_data else None
            test_dataset = VTRDatasetOCR(test_data, ratio=vtr.ratio, *dataset_args) if test_data else None

            char2int_dict = {char: i + 1 for i, char in enumerate(char2array.keys())}
            logger.info(
                f"OCR parameters: hidden size: {vtr.hidden_size_ocr}, # layers: {vtr.num_layers_ocr}, "
                f"# classes: {len(char2array.keys())}"
            )
            ocr = OCRHead(
                input_size=vtr.out_channels * (vtr.font_size // vtr.pool_kernel_size ** (len(channels) - 1)),
                hidden_size=vtr.hidden_size_ocr,
                num_layers=vtr.num_layers_ocr,
                num_classes=len(char2array.keys()),
            )

            model = SequenceClassifier(model_config, embedder, training_config.max_seq_len, char2int_dict, ocr,
                                       vtr.alpha)

    else:
        train_dataset = TokenizedIterableDataset(train_dataset, nllb_tokenizer, training_config.max_seq_len)
        val_dataset = TokenizedDataset(val_dataset, nllb_tokenizer, training_config.max_seq_len)
        test_dataset = TokenizedDataset(test_dataset, nllb_tokenizer, training_config.max_seq_len)

        embedder = TTREmbedder(train_dataset.tokenizer.vocab_size, model_config.emb_size)

    #model_config.num_classes = train_dataset.get_num_classes()
    #model = SequenceClassifier(model_config, embedder, training_config.max_seq_len)

    train(
        model,
        train_dataset,
        training_config,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        ocr_flag=not args.no_ocr,
    )


if __name__ == "__main__":
    logger.info("Loading data...")
    args = configure_arg_parser().parse_args()
    logger.info("Loading data...")
    train_data = load_json(args.train_data)
    val_data = load_json(args.val_data) if args.val_data else None
    test_data = load_json(args.test_data) if args.test_data else None
    train_vtr_encoder(args, train_data, val_data, test_data)
