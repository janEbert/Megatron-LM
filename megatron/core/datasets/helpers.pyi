import numpy


def build_mapping(
        docs_: numpy.ndarray,
        sizes_: numpy.ndarray,
        num_epochs: int,
        max_num_samples: int,
        max_seq_length: int,
        short_seq_prob: float,
        seed: int,
        verbose: bool,
        min_num_sent: int,
) -> numpy.ndarray:
    ...


def build_blocks_mapping(
        docs_: numpy.ndarray,
        sizes_: numpy.ndarray,
        titles_sizes_: numpy.ndarray,
        num_epochs: int,
        max_num_samples: int,
        max_seq_length: int,
        seed: int,
        verbose: bool,
        use_one_sent_blocks: bool,
) -> numpy.ndarray:
    ...


def build_sample_idx_int32(
        sizes_: numpy.ndarray,
        document_idx_: numpy.ndarray,
        seq_length: int,
        num_epochs: int,
        tokens_per_epoch: int,
        drop_last_partial_sequence: bool = True,
        add_extra_token_to_sequence: int = 1,
) -> numpy.ndarray:
    ...


def build_sample_idx_int64(
        sizes_: numpy.ndarray,
        document_idx_: numpy.ndarray,
        seq_length: int,
        num_epochs: int,
        tokens_per_epoch: int,
        drop_last_partial_sequence: bool = True,
        add_extra_token_to_sequence: int = 1,
) -> numpy.ndarray:
    ...


def build_blending_indices(
        dataset_index: numpy.ndarray,
        dataset_sample_index: numpy.ndarray,
        weights: numpy.ndarray,
        num_datasets: int,
        size: int,
        verbose: bool,
) -> None:
    ...


def build_exhaustive_blending_indices(
        dataset_index: numpy.ndarray,
        dataset_sample_index: numpy.ndarray,
        sizes: numpy.ndarray,
        num_datasets: int,
) -> None:
    ...


def build_sample_idx(
    sizes: numpy.ndarray,
    document_indices: numpy.ndarray,
    sequence_length: int,
    num_epochs: int,
    tokens_per_epoch: int,
    drop_last_partial_sequence: bool = True,
    add_extra_token_to_sequence: bool = True,
):
    ...
