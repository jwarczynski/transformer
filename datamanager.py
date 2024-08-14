import os

import datasets
from datasets import load_dataset
from torch.utils.data import Sampler, DataLoader
import random
import math


class BucketBatchSampler(Sampler):
    def __init__(self, datasets, batch_size, shuffle=True):
        """
        datasets: List of individual bucket datasets.
        batch_size: Desired batch size.
        shuffle: Whether to shuffle the data within each bucket.
        """
        super().__init__()
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_indices = self._prepare_indices()

    def _prepare_indices(self):
        all_indices = []
        start_idx = 0
        for dataset in self.datasets:
            dataset_size = len(dataset)
            indices = list(range(start_idx, start_idx + dataset_size))
            if self.shuffle:
                random.shuffle(indices)
            # Split indices into batches
            batch_indices = [indices[i:i + self.batch_size] for i in range(0, dataset_size, self.batch_size)]
            all_indices.extend(batch_indices)
            start_idx += dataset_size
        if self.shuffle:
            random.shuffle(all_indices)  # Shuffle the order of batches from different buckets
        return all_indices

    def __iter__(self):
        for batch in self.bucket_indices:
            yield batch

    def __len__(self):
        return sum(math.ceil(len(dataset) / self.batch_size) for dataset in self.datasets)


class DataPreprocessing:
    def __init__(self, tokenizer, batch_size, max_length, dataset_kwargs, kwargs, train_size=None, ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_kwargs = dataset_kwargs
        self.train_size = train_size
        self.kwargs = kwargs
        self.batch_size = batch_size

        self.splits = {}
        self.buckets = {}

    def __call__(self, split: str, size: int | None = None, seed: int | None = None):
        if split not in self.splits:
            self.load_and_bucket_split(split, size, seed)

        return self.splits[split], self.buckets[split]

    def load_and_bucket_split(self, split, size=None, seed=None):
        split_data = self.load_split(split, size, seed)
        split_data = split_data.flatten()
        split_data = split_data.map(
            lambda x: {'de_sentence_len': len(x['translation.de'].split())}, num_proc=os.cpu_count()
        ).sort('de_sentence_len')

        split_data = split_data.map(
            lambda batch: self.tokenize(batch, self.tokenizer),
            batched=True, batch_size=self.batch_size, remove_columns=['translation.de', 'translation.en']
        )

        split_data.set_format('torch', columns=['input_ids_en', 'input_ids_de'])
        self.splits[split] = split_data
        self.buckets[split] = self.bucket(split_data, self.batch_size)

    @staticmethod
    def tokenize(batch, tokenizer):
        tokenized_batch_de = tokenizer(
            batch['translation.de'], padding=True, truncation=True, max_length=512, add_special_tokens=True
        )['input_ids']
        padded_batch_de = [example + [tokenizer.pad_token_id] for example in tokenized_batch_de]
        return {
            'input_ids_de': padded_batch_de,
            'input_ids_en': tokenizer(batch['translation.en'], padding=True, truncation=True, max_length=512,
                                      add_special_tokens=False)['input_ids'],
            'len_de': [len(example) for example in tokenized_batch_de]
        }

    def load_split(self, split: str, size: int = None, seed: int = None):
        split_data = load_dataset(**self.dataset_kwargs, split=split)
        if size is not None:
            split_data = split_data.shuffle(seed=seed).select(range(size))
        return split_data

    def bucket(self, split: datasets.Dataset, batch_size: int):
        return [split.select(range(i, min((i + batch_size), len(split)))) for i in range(0, len(split), batch_size)]


def create_dataloader(tokenizer, batch_size, size, split='train', max_length=512, seed=None, shuffle=True):
    dataset_kwargs = {
        'path': 'wmt14',
        'name': 'de-en'
    }
    dataset_manager = DataPreprocessing(tokenizer, batch_size, max_length, dataset_kwargs, kwargs={})
    split_data, buckets = dataset_manager(split, size=1000)

    sampler = BucketBatchSampler(buckets, batch_size, shuffle=shuffle)
    dataloader = DataLoader(split_data, batch_sampler=sampler)
    return dataloader


def get_dataloader(data_manager: DataPreprocessing, split: str, size: int = None):
    split_data, buckets = data_manager(split, size=size)
    sampler = BucketBatchSampler(buckets, data_manager.batch_size, shuffle=True)
    return DataLoader(split_data, batch_sampler=sampler)
