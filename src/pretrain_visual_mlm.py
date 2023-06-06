from src.data_sets.vtr_dataset import VTRDatasetOCR
from src.models.pretraining import MaskedVisualLM
from src.models.vtr.ocr import OCRHead

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
)
from src.data_sets.translation_datasets import FloresDataset, NLLBDatasetRuEn
from torch.utils.data import IterableDataset, Dataset
from src.utils.augmentation import init_augmentations
from src.utils.config import TransformerConfig, TrainingConfig, VTRConfig, AugmentationConfig
from src.utils.pretrain import train


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()

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

    arg_parser.add_argument("--dataset-dir", type=str, help="Directory of validation and test Flores files.")

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


def pretrain_vtr(args: Namespace):
    logger.info("Pre-training masked language model for VTR.")
    model_config = TransformerConfig.from_arguments(args)
    training_config = TrainingConfig.from_arguments(args)
    vtr = VTRConfig.from_arguments(args)
    augmentation_config = AugmentationConfig.from_arguments(args)

    with open(args.probas, "rb") as f:
        probas = pickle.load(f)

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

    nllb_tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-3.3B")

    train_dataset: IterableDataset | Dataset
    val_dataset: Dataset
    test_dataset: Dataset

    with open("resources/perc_ru_en.pkl", "rb") as f:
        perc_ru_en = pickle.load(f)
    train_dataset = NLLBDatasetRuEn(probas=perc_ru_en)
    lang2label = train_dataset.get_lang2label()

    if args.no_augmentation:
        val_dataset = FloresDataset(lang2label=lang2label, split="dev")
        test_dataset = FloresDataset(lang2label=lang2label, split="devtest")
    else:
        if not args.dataset_dir:
            logger.error("Need directory with augmented val and test Flores")
            return

        logger.info(
            f"Noisy dataset: expected_changes_per_word:{augmentation_config.expected_changes_per_word}, proba_per_text:{augmentation_config.proba_per_text}, expected_changes_per_text:{augmentation_config.expected_changes_per_text}, max_augmentations={augmentation_config.max_augmentations}"
        )

        train_dataset = AugmentationDataset(
            dataset=train_dataset,
            augmentations=augmentations,
            proba_per_text=augmentation_config.proba_per_text,
            expected_changes_per_text=augmentation_config.expected_changes_per_text,
            max_augmentations=augmentation_config.max_augmentations,
        )
        val_dataset = FloresDataset(lang2label, split="dev", dataset_dir=args.dataset_dir)
        test_dataset = FloresDataset(lang2label, split="devtest", dataset_dir=args.dataset_dir)

    dataset_args = (char2array, vtr.window_size, vtr.stride, training_config.max_seq_len)
    if args.no_ocr:
        train_dataset = SlicesIterableDataset(train_dataset, char2array)
        val_dataset = SlicesDataset(val_dataset, char2array)
        test_dataset = SlicesDataset(test_dataset, char2array)

        model = MaskedVisualLM(
            model_config.n_head,
            model_config.num_layers,
            model_config.dropout,
            vtr.font_size,
            vtr.window_size,
            model_config.emb_size,
            vtr.no_verbose,
            vtr.save_plots,
            training_config.random_state,
        )
    else:
        train_dataset = VTRDatasetOCR(train_data, ratio=vtr.ratio, *dataset_args)
        val_dataset = VTRDatasetOCR(val_data, ratio=vtr.ratio, *dataset_args) if val_data else None
        test_dataset = VTRDatasetOCR(test_data, ratio=vtr.ratio, *dataset_args) if test_data else None

        charset = train_dataset.char_set | val_dataset.char_set | test_dataset.char_set
        char2int = {char: i + 1 for i, char in enumerate(charset)}
        # char2int = {char: i + 1 for i, char in enumerate(char2array.keys())}

        logger.info(
            f"OCR parameters: hidden size: {vtr.hidden_size_ocr}, # layers: {vtr.num_layers_ocr}, "
            f"# classes: {len(charset)}"  # char2array.keys()
        )
        ocr = OCRHead(
            input_size=vtr.font_size,
            hidden_size=vtr.hidden_size_ocr,
            num_layers=vtr.num_layers_ocr,
            # num_classes=len(char2array.keys()),
            num_classes=len(charset),
        )

        model = MaskedVisualLM(
            model_config.n_head,
            model_config.num_layers,
            model_config.dropout,
            vtr.font_size,
            vtr.window_size,
            model_config.emb_size,
            vtr.no_verbose,
            vtr.save_plots,
            vtr.plot_every,
            training_config.random_state,
            ocr,
            char2int,
            vtr.alpha,
        )

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
    _args = configure_arg_parser().parse_args()
    pretrain_vtr(_args)
