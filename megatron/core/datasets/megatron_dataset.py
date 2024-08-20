# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import hashlib
import json
from abc import ABC, abstractmethod
from collections import OrderedDict
import os
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy
import torch

from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.utils import Split

LowLevelDataset = Union[IndexedDataset, Iterable]


class MegatronDataset(ABC, torch.utils.data.Dataset):
    """The highest level wrapper class from which all dataset classes should inherit

    Args:
        dataset (LowLevelDataset): The dataset around which to build the MegatronDataset

        dataset_path (Optional[str]): The real path on disk to the dataset, for bookkeeping

        indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (Optional[int]): The minimum number of samples to build from the indexed dataset. When None, build as many samples as correspond to one epoch.

        index_split (Split): The indices Split

        config (BlendedMegatronDatasetConfig): The config
    """

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: numpy.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: BlendedMegatronDatasetConfig,
    ) -> None:
        self.dataset = dataset
        self.dataset_path = dataset_path

        path_to_cache = config.path_to_cache
        indices_hash = hashlib.md5(indices).hexdigest()
        if path_to_cache:
            self.path_to_cache_indices = os.path.join(
                path_to_cache,
                f"{os.path.basename(self.dataset_path)}_super_document_indices_{indices_hash}.npy",
            )
            if not os.path.isfile(self.path_to_cache_indices):
                os.makedirs(os.path.dirname(self.path_to_cache_indices), exist_ok=True)
                numpy.save(
                    self.path_to_cache_indices,
                    indices,
                    allow_pickle=True,
                )
            self.indices = numpy.load(self.path_to_cache_indices, allow_pickle=True, mmap_mode='r')
        else:
            self.indices = indices
        self.num_samples = num_samples
        self.index_split = index_split
        self.config = config

        self.unique_identifiers = OrderedDict()

        self.unique_identifiers["class"] = type(self).__name__
        self.unique_identifiers["dataset_path"] = self.dataset_path
        self.unique_identifiers["num_samples"] = self.num_samples
        self.unique_identifiers["index_split"] = self.index_split.name
        for attr in self._key_config_attributes():
            self.unique_identifiers[attr] = getattr(self.config, attr)

        self.unique_description = json.dumps(
            self.unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers
        )
        self.unique_description_hash = hashlib.md5(
            self.unique_description.encode("utf-8")
        ).hexdigest()

    def __getstate__(self) -> Dict[str, Any]:
        """Get the state during pickling

        Returns:
            Dict[str, Any]
        """
        state = self.__dict__.copy()
        if self.config.path_to_cache:
            state.pop('indices')
        return state

    def __setstate__(self, state: Dict[str, Any]):
        """Set the state during un-pickling

        Args:
            state (Dict[str, Any]): The state dict
        """
        for (k, v) in state.items():
            self.__dict__[k] = v
        if self.config.path_to_cache:
            assert os.path.isfile(self.path_to_cache_indices), \
                "data is not present in cache; cannot load pickle."
            self.indices = numpy.load(self.path_to_cache_indices, allow_pickle=True, mmap_mode='r')

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        """Return the number of elements in the underlying low level dataset for the purpose of
        segregating the train/valid/test split indices

        It may be that the low level dataset can be split any number of ways, depending on the mid
        level dataset it supports, which is why we define the "number of elements" function
        separately from the __len__ function here in the mid level dataset class

        Args:
            low_level_dataset (LowLevelDataset): The underlying low level dataset

        Returns:
            int: The number of elements in the underlying low level dataset
        """
        raise NotImplementedError

    @staticmethod
    def build_low_level_dataset(
        dataset_path: str, config: BlendedMegatronDatasetConfig
    ) -> LowLevelDataset:
        """Build the low level dataset via a function to be called from within
        BlendedMegatronDatasetBuilder.build_generic_dataset

        It may be that the low level dataset spans any subset of train/valid/test splits, which is
        why we define a static "build" function separately from the constructor in the mid level
        dataset class

        Args:
            dataset_path (str): The real path on disk to the dataset

            config (BlendedMegatronDatasetConfig): The dataset config

        Returns:
            LowLevelDataset: The low level dataset
        """
        raise NotImplementedError

    @staticmethod
    def _key_config_attributes() -> List[str]:
        """Return all config attributes which contribute to uniquely identifying the dataset.

        These attributes will be used to build a uniquely identifying string and MD5 hash which
        will be used to cache/load dataset resources from run to run.

        Returns:
            List[str]: The key config attributes
        """
        return ["random_seed", "sequence_length", "split", "split_matrix", "tokenizer"]

    @abstractmethod
    def __len__(self) -> int:
        """Return the length of the dataset

        Returns:
            int: See abstract implementation
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, numpy.ndarray]]:
        """Return from the dataset

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, Union[torch.Tensor, numpy.ndarray]]: See abstract implementation
        """
        pass
