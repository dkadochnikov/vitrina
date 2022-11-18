from collections import defaultdict
from typing import Any, Dict, List, Union

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from utils.slicer import VTRSlicer
from utils.utils import clean_text


class VTRDatasetSL(Dataset):
    labeled_texts = []

    def __init__(
        self,
        labeled_texts: List[Dict[str, Union[List[List[Union[str, int]]], int]]],
        font: str,
        font_size: int = 15,
        window_size: int = 30,
        stride: int = 5,
        max_seq_len: int = 512,
        max_slices_count_per_word: int = None,
    ):
        self.labeled_texts = labeled_texts
        self.slicer = VTRSlicer(
            font=font, font_size=font_size, window_size=window_size, stride=stride
        )
        self.max_seq_len = max_seq_len
        self.max_slices_count_per_word = max_slices_count_per_word
        self.font_size = font_size
        self.window_size = window_size
        self.test_dataset = False

    def __len__(self) -> int:
        return len(self.labeled_texts)

    def __getitem__(self, index) -> Dict[str, Any]:
        labeled_text = self.labeled_texts[index]

        slices = []
        labels = []

        for word, label in labeled_text["text"][:56]:
            cleaned_word = clean_text(word)
            if not cleaned_word:
                continue

            word_slices = self.slicer(cleaned_word, self.max_slices_count_per_word)

            slices.append(word_slices)
            labels.append(label)

        slices = slices[: self.max_seq_len]
        labels = labels[: self.max_seq_len]

        return {
            "slices": slices,
            "labels": labels,
        }

    def collate_function(
        self, input_batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Union[torch.Tensor, int]]:
        key2values = defaultdict(list)
        max_word_len_in_slices = 0
        max_seq_len = 0
        for example in input_batch:
            slices = example["slices"]
            if len(slices) != 0:
                max_word_len_in_slices = max(
                    max_word_len_in_slices, max(map(len, slices))
                )
                max_seq_len = max(max_seq_len, len(slices))

            key2values["slices"].append(slices)
            key2values["labels"].append(example["labels"])

        zero_word = torch.zeros(
            (max_word_len_in_slices, self.font_size, self.window_size)
        )

        result = defaultdict(list)
        for text, labels in zip(key2values["slices"], key2values["labels"]):
            padding_in_word_count = max_seq_len - len(text)

            padded_words = []
            tokens_mask = []
            for word in text:
                padding_size = max_word_len_in_slices - len(word)
                padded_words.append(
                    F.pad(
                        word,
                        (0, 0, 0, 0, 0, padding_size),
                        mode="constant",
                        value=0,
                    )
                )
                tokens_mask.extend([1] * len(word) + [0] * padding_size)

            result["slices"].append(padded_words + [zero_word] * padding_in_word_count)
            result["tokens_mask"].append(
                tokens_mask
                + [0] * max_word_len_in_slices * padding_in_word_count
            )
            result["labels"].append(labels + [-1] * padding_in_word_count)
            result["words_mask"].append(
                [1] * len(text) + [0] * padding_in_word_count
            )

        torch_result = {
            "slices": torch.stack([torch.cat(text) for text in result["slices"]]),
            "max_word_len": max_word_len_in_slices,
            "words_mask": torch.tensor(result["words_mask"]),
            "tokens_mask": torch.tensor(result["tokens_mask"]),
            "labels": torch.tensor(result["labels"]),
        }

        return torch_result


if __name__ == "__main__":
    labeled_texts = [{"text": [["|{ 0шk@", 0], ["н@", 0], ["0|<ошке", 0]], "label": 0}, {"text": [["длинношеее", 0], ["животное", 0]], "label": 0}]
    dataset = VTRDatasetSL(labeled_texts, "fonts/NotoSans.ttf")
    data_loader = DataLoader(
        dataset, batch_size=2, collate_fn=dataset.collate_function
    )
    batch = next(iter(data_loader))
    print(batch)
