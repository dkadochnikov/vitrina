from collections import defaultdict

import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import Dataset
from transformers import BertTokenizer

from src.data_sets.common import SLDatasetSample
from src.utils.common import clean_text


class BERTDatasetSL(Dataset):
    def __init__(
        self,
        labeled_texts: list[SLDatasetSample],
        tokenizer: str,
        max_seq_len: int = 512,
    ):
        logger.info(f"Initializing BERTDatasetSL with {len(labeled_texts)} samples, use max seq len {max_seq_len}")
        self.labeled_texts = labeled_texts
        self.max_seq_len = max_seq_len

        logger.info(f"Loading tokenizer from '{tokenizer}'")
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)

    def __len__(self) -> int:
        return len(self.labeled_texts)

    def __getitem__(self, index: int) -> dict[str, list]:
        labeled_text: list[tuple[str, int]] = self.labeled_texts[index]["text"]

        encoded_words = []
        labels = []

        for word, label in labeled_text:
            cleaned_word = clean_text(word)
            if not cleaned_word:
                continue
            encoded_dict = self.tokenizer(
                cleaned_word,
                add_special_tokens=False,
                max_length=self.max_seq_len,
                return_attention_mask=False,
                return_token_type_ids=False,
                truncation=True,
            )
            encoded_word = encoded_dict["input_ids"]
            encoded_words.append(encoded_word)
            labels.append(label)

        return {"words_input_ids": encoded_words, "labels": labels}

    def collate_function(self, batch: list[dict[str, list]]) -> dict[str, torch.Tensor]:
        key2values: dict[str, list[torch.Tensor]] = defaultdict(list)
        max_seq_len = 0
        max_word_len = 0
        for item in batch:
            word_input_ids = item["words_input_ids"]
            max_seq_len = max(max_seq_len, len(word_input_ids))
            for word in word_input_ids:
                max_word_len = max(max_word_len, len(word))
            key2values["words_input_ids"].append(torch.tensor(word_input_ids))

            key2values["labels"].append(torch.tensor(item["labels"]))

        pad_token_id = self.tokenizer.pad_token_id

        max_seq_len_in_words = self.max_seq_len // max_word_len
        max_seq_len_in_tokens = max_seq_len_in_words * max_word_len

        batch_labels = []
        for text, labels in zip(key2values["words_input_ids"], key2values["labels"]):
            text_attention_mask = []
            words = []
            for word in text:
                padding_size = max_word_len - len(word)
                words.append(
                    F.pad(
                        torch.tensor(word),
                        (0, padding_size),
                        mode="constant",
                        value=pad_token_id,
                    )
                )
                text_attention_mask.extend([1] * len(word) + [0] * padding_size)

            if len(words) > 0:
                key2values["input_ids"].append(torch.cat(words)[:max_seq_len_in_tokens])
                key2values["attention_mask"].append(torch.tensor(text_attention_mask)[:max_seq_len_in_tokens])
                batch_labels.append(labels[:max_seq_len_in_words])

        return {
            "max_word_len": torch.tensor(max_word_len, dtype=torch.int32),
            "input_ids": torch.nn.utils.rnn.pad_sequence(key2values["input_ids"], batch_first=True).long(),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(key2values["attention_mask"], batch_first=True).long(),
            "labels": torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-1).long(),
        }
