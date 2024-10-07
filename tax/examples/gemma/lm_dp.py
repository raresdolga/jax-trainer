"""Datasets for language modelling
"""

from typing import Dict, Iterable
import os
from os import PathLike
from pathlib import Path
from tqdm import tqdm
import torch
from numpy.typing import NDArray
import numpy as np
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer


class WindowDataset2(Dataset):
    def __init__(self, path: PathLike, window_size: int):
        assert window_size > 1, f"Invalid window_size: {window_size}"

        self._path = path
        self._tokens = None
        self._window_size = window_size
        self.mem_dtype = np.int32

    def _get_tokens(self) -> NDArray:
        # Lazily mmap the file with tokens
        if self._tokens is None:
            self._tokens = np.memmap(self._path, mode="r", dtype=self.mem_dtype)
            assert (
                len(self._tokens.shape) == 1 and self._tokens.dtype == self.mem_dtype
            ), (f"Invalid shape {self._tokens.shape}",)
        return self._tokens

    def __len__(self) -> int:
        return len(self._get_tokens()) - self._window_size - 1

    def __getitem__(self, window_index: int) -> Dict[str, NDArray]:
        tokens = self._get_tokens()
        inputs = tokens[window_index : window_index + self._window_size]
        targets = tokens[window_index : window_index + self._window_size]
        return {"input_ids": inputs, "labels": targets}

    def __getitems__(self, window_indexs: int) -> Dict[str, NDArray]:
        batch = [self.__getitem__(i) for i in window_indexs]
        return batch

    @staticmethod
    def collate_np(datapoints: Iterable[Dict[str, NDArray]]) -> Dict[str, NDArray]:
        return {
            "input_ids": np.stack([datapoint["input_ids"] for datapoint in datapoints]),
            "labels": np.stack([datapoint["labels"] for datapoint in datapoints]),
        }

    @staticmethod
    def collate_torch(datapoints: Iterable[Dict[str, NDArray]]) -> Dict[str, NDArray]:
        return {
            "input_ids": torch.stack(
                [
                    torch.tensor(datapoint["input_ids"], dtype=torch.long)
                    for datapoint in datapoints
                ]
            ),
            "labels": torch.stack(
                [
                    torch.tensor(datapoint["labels"], dtype=torch.long)
                    for datapoint in datapoints
                ]
            ),
        }


class TinyStories:
    """Group examples such that padding is reduced to minimal"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        cache_dir: str,
        num_load_procs: int,
        max_seq_len: int,
    ) -> None:
        self._cache_dir = cache_dir
        self._tokenizer = tokenizer
        self._num_load_procs = num_load_procs
        self._max_seq_len = max_seq_len
        self.mem_dtype = np.int32

    def get_collate_fn(self, return_type="torch", **kwargs):
        if return_type == "torch":
            return WindowDataset2.collate_torch
        else:
            return WindowDataset2.collate_np

    @property
    def tokenizer(self):
        return self._tokenizer

    def _tokenize(self, elements):
        elements = self._tokenizer(
            elements["text"],
            return_special_tokens_mask=False,
            add_special_tokens=False,  # True,
            return_attention_mask=False,
            truncation=False,
            return_length=True,
        )
        # TODO: this is a bit slow because add_special_tokens
        # does not work with self added tokens - investigate
        elements["input_ids"] = [
            [self._tokenizer.bos_token_id] + elements["input_ids"][i]
            for i in range(len(elements["input_ids"]))
        ]
        # print(elements["length"])
        elements["length"] = [
            1 + elements["length"][i] for i in range(len(elements["length"]))
        ]
        return elements

    def tokenize_and_memmap(self, raw_dataset):
        # pre-tokenize val data
        tokenized = raw_dataset.map(
            self._tokenize,
            batched=True,
            num_proc=self._num_load_procs,
            batch_size=10000,
            remove_columns=["text"],
            desc="Tokenizing Tiny Stories dataset",
            load_from_cache_file=False,  # extra preproc time to save disk
        )
        dataset_path = Path(self._cache_dir)
        dataset_path.mkdir(parents=True, exist_ok=True)

        for split, tokens in tokenized.items():
            filename = dataset_path / f"{split}.bin"

            assert self._tokenizer.vocab_size < 2**32

            dtype = (
                self.mem_dtype
            )  # (can do since enc.max_token_value == 50256 is < 2**16)
            array_len = np.sum(tokens["length"], dtype=self.mem_dtype)
            array = np.memmap(filename, dtype=dtype, mode="w+", shape=(array_len,))

            total_batches, index = 1024, 0
            for batch_index in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = tokens.shard(
                    num_shards=total_batches,
                    index=batch_index,
                    contiguous=True,
                ).with_format("numpy")
                array_batch = np.concatenate(batch["input_ids"])
                # Write into mmap
                array[index : index + len(array_batch)] = array_batch
                index += len(array_batch)

            array.flush()

        return tokenized

    def tokenize(self, raw_dataset, force_preproc):
        cond = (
            (not os.path.exists(Path(self._cache_dir) / "train.bin"))
            or (not os.path.exists(Path(self._cache_dir) / "validation.bin"))
            or force_preproc
        )
        if cond:
            self.tokenize_and_memmap(raw_dataset)
        return {
            "train": WindowDataset2(
                path=Path(self._cache_dir) / "train.bin", window_size=self._max_seq_len
            ),
            "validation": WindowDataset2(
                path=Path(self._cache_dir) / "validation.bin",
                window_size=self._max_seq_len,
            ),
        }

    def get_raw_data(self):
        os.makedirs(self._cache_dir, exist_ok=True)
        dataset = load_dataset(
            "roneneldan/TinyStories",
            cache_dir=self._cache_dir,
            num_proc=self._num_load_procs,
        )
        assert dataset.keys() == {"train", "validation"}

        # dataset["train"] = dataset["train"].select(np.arange(10240))
        # dataset["validation"] = dataset["validation"].select(np.arange(10240))
        return dataset


if __name__ == "__main__":
    # pdm run python3 -m latte_trans.preproc.tiny_stories
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    base_dir = "/data_rares/data"
    MAX_SEQ_LEN = 1024
    cache_dir = os.path.join(base_dir, "input", "tiny-sories")
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2-2b",
        cache_dir=Path(base_dir) / "input/cache_hugg",
        truncation_side="right",
        padding_side="right",
    )

    dp = TinyStories(
        tokenizer=tokenizer,
        cache_dir=cache_dir,
        max_seq_len=MAX_SEQ_LEN,
        num_load_procs=max(1, os.cpu_count() - 10),
    )
    raw_data = dp.get_raw_data()

    dataset = dp.tokenize(raw_data, force_preproc=False)

    # import multiprocessing
    dl = DataLoader(
        dataset=dataset["train"],
        batch_size=10,
        shuffle=True,
        collate_fn=WindowDataset2.collate_np,
    )
    print(next(iter(dl)))
