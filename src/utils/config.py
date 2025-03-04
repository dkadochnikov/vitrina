from argparse import Namespace, ArgumentParser
from dataclasses import dataclass, fields


@dataclass
class VTRConfig:
    font: str
    font_size: int
    window_size: int
    stride: int
    conv_kernel_size: int
    pool_kernel_size: int
    out_channels: int
    hidden_size_ocr: int
    num_layers_ocr: int
    ratio: float
    alpha: float
    max_slices_count_per_word: int = None
    no_verbose: bool = False
    save_plots: bool = False
    plot_every: int = 100

    @classmethod
    def from_arguments(cls, args: Namespace) -> "VTRConfig":
        config_fields = [it.name for it in fields(cls)]
        kwargs = {it: getattr(args, it) for it in config_fields}
        return cls(**kwargs)

    @classmethod
    def add_to_arg_parser(cls, arg_parser: ArgumentParser) -> ArgumentParser:
        arg_parser.add_argument(
            "--font", type=str, default="resources/fonts/NotoSans.ttf", help="Path to font that used for rendering."
        )
        arg_parser.add_argument("--font-size", type=int, default=16, help="Font size to use for rendering.")

        arg_parser.add_argument("--window-size", type=int, default=32, help="Window size to slice the image w/ text.")
        arg_parser.add_argument("--stride", type=int, default=5, help="Window step size.")

        arg_parser.add_argument("--conv-kernel-size", type=int, default=3, help="Kernel size to use for convolutions.")
        arg_parser.add_argument("--pool-kernel-size", type=int, default=2, help="Kernel size to use for pooling.")
        arg_parser.add_argument(
            "--out-channels", type=int, default=256, help="Number of output channels in the last convolutional layer."
        )
        arg_parser.add_argument("--hidden-size-ocr", type=int, default=256, help="Hidden size to use for OCR.")
        arg_parser.add_argument("--num-layers-ocr", type=int, default=2, help="Number of layers to use in OCR.")
        arg_parser.add_argument("--ratio", type=float, default=0.7, help="Ratio of letter to be detected on a slice.")
        arg_parser.add_argument("--alpha", type=float, default=1, help="Coefficient for losses summation")
        arg_parser.add_argument(
            "--max-slices-count-per-word", type=int, default=9, help="Maximum number of slices per word."
        )
        arg_parser.add_argument("--no-verbose", action="store_true", help="Build plots for pre-training.")
        arg_parser.add_argument("--save-plots", action="store_true", help="Save plots.")
        arg_parser.add_argument("--plot-every", type=int, help="Plot patches every N steps.")
        return arg_parser


@dataclass
class TransformerConfig:
    num_layers: int
    emb_size: int
    n_head: int
    dropout: float
    num_classes: int

    @classmethod
    def from_arguments(cls, args: Namespace) -> "TransformerConfig":
        config_fields = [it.name for it in fields(cls)]
        kwargs = {it: getattr(args, it) for it in config_fields}
        return cls(**kwargs)

    @classmethod
    def add_to_arg_parser(cls, arg_parser: ArgumentParser) -> ArgumentParser:
        arg_parser.add_argument("--num-layers", type=int, default=8, help="Number of layers in encoder.")
        arg_parser.add_argument("--emb-size", type=int, default=768, help="Embedding size.")
        arg_parser.add_argument("--n-head", type=int, default=12, help="Number of heads in MHA layers.")
        arg_parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
        arg_parser.add_argument("--num-classes", type=int, default=2, help="Number of labels' classes.")
        return arg_parser


@dataclass
class TrainingConfig:
    max_seq_len: int = 512
    batch_size: int = 32
    steps: int = 10000
    p: float = None

    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup: int = 1000
    beta1: float = 0.9
    beta2: float = 0.999

    device: str = None
    random_state: int = 21
    log_every: int = 1000
    save_every: int = 20000
    num_workers: int = 1

    no_average: bool = False

    checkpoint_path: str = None

    @classmethod
    def from_arguments(cls, args: Namespace) -> "TrainingConfig":
        config_fields = [it.name for it in fields(cls)]
        kwargs = {it: getattr(args, it) for it in config_fields}
        return cls(**kwargs)

    @classmethod
    def add_to_arg_parser(cls, arg_parser: ArgumentParser) -> ArgumentParser:
        arg_parser.add_argument("--max-seq-len", type=int, default=512, help="Maximum len of tokens per sequence.")
        arg_parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
        arg_parser.add_argument("--steps", type=int, default=10000, help="Number of training steps.")
        arg_parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
        arg_parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay parameter.")
        arg_parser.add_argument("--warmup", type=int, default=1000, help="Number of warmup steps.")
        arg_parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for Adam.")
        arg_parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for Adam.")
        arg_parser.add_argument(
            "--device", type=str, default=None, help="'cuda', or 'cpu'. If 'None' it will be set automatically."
        )
        arg_parser.add_argument("--random-state", type=int, default=21, help="Random state.")
        arg_parser.add_argument("--log-every", type=int, default=1000, help="Log every N steps.")
        arg_parser.add_argument("--save-every", type=int, default=20000, help="Save model every N steps.")
        arg_parser.add_argument("--num-workers", type=int, default=1, help="Number of workers for data loaders.")
        arg_parser.add_argument(
            "--no-average", action="store_true", help="Do not use averaging for evaluation metrics."
        )
        arg_parser.add_argument("--checkpoint-path", type=str, help="Path to checkpoint.")
        arg_parser.add_argument("--p", type=float, default=None, help="Ratio of toxic samples.")
        return arg_parser


@dataclass
class AugmentationConfig:
    leet: str
    clusters: str
    expected_changes_per_word: int = 2
    expected_changes_per_text: int = 3
    max_augmentations: int = 2
    proba_per_text: float = 0.8

    @classmethod
    def from_arguments(cls, args: Namespace) -> "AugmentationConfig":
        config_fields = [it.name for it in fields(cls)]
        kwargs = {it: getattr(args, it) for it in config_fields}
        return cls(**kwargs)

    @classmethod
    def add_to_arg_parser(cls, arg_parser: ArgumentParser) -> ArgumentParser:
        arg_parser.add_argument(
            "--expected-changes-per-word",
            type=int,
            default=2,
            help="Expected value of words in every text that we want to make noisy.",
        )
        arg_parser.add_argument(
            "--expected-changes-per-text",
            type=int,
            default=3,
            help="Expected value of chars in a word that we want to make noisy.",
        )
        arg_parser.add_argument(
            "--max-augmentations",
            type=int,
            default=2,
            help="Maximum value of augmentations that can be applied to every word.",
        )

        arg_parser.add_argument(
            "--proba-per-text",
            type=float,
            default=0.8,
            help="Probability of text augmentation.",
        )

        arg_parser.add_argument(
            "--leet",
            type=str,
            default="resources/nllb/letter_replacement/leet.json",
            help="Path to leet symbols,",
        )

        arg_parser.add_argument(
            "--clusters",
            type=str,
            default="resources/nllb/letter_replacement/clusterization.pkl",
            help="Path to cluster symbols.",
        )

        return arg_parser
