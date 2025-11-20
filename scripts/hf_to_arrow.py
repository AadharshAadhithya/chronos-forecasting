from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from datasets import load_dataset
from gluonts.dataset.arrow import ArrowWriter
from tqdm import tqdm


def convert_to_arrow(
    path: Union[str, Path],
    time_series: Sequence[np.ndarray],
    start_times: Optional[Sequence[pd.Period | np.datetime64]] = None,
    compression: str = "lz4",
) -> None:
    """
    Store a collection of time series into a GluonTS-compatible Arrow file.
    """
    if start_times is None:
        start_times = [np.datetime64("2000-01-01 00:00", "s")] * len(time_series)

    assert len(time_series) == len(start_times), "Mismatch between data and start times"

    dataset = [
        {
            "start": start,
            "target": np.asarray(series, dtype=np.float32),
        }
        for series, start in zip(time_series, start_times)
    ]

    ArrowWriter(compression=compression).write_to_file(dataset, path=path)


def hf_dataset_to_arrow(
    hf_repo: str,
    hf_config: str,
    output_path: Union[str, Path],
    freq: str = "D",
    max_series: Optional[int] = None,
    compression: str = "lz4",
    streaming: bool = False,
) -> None:
    """
    Download a Hugging Face dataset and convert it into a single Arrow file.
    """
    ds = load_dataset(hf_repo, hf_config, split="train", streaming=streaming)

    time_series: List[np.ndarray] = []
    start_times: List[pd.Period] = []

    iterator: Iterable = ds if streaming else range(len(ds))

    for idx in tqdm(iterator, desc="Converting"):
        if streaming:
            entry = idx  # idx is the actual element in streaming mode
        else:
            entry = ds[idx]

        if max_series is not None and len(time_series) >= max_series:
            break

        timestamps = entry.get("timestamp")
        if timestamps is not None and len(timestamps) > 0:
            start = pd.Period(timestamps[0], freq=freq)
        else:
            start = pd.Period("2000-01-01 00:00", freq=freq)

        start_times.append(start)
        time_series.append(np.asarray(entry["target"], dtype=np.float32))

    convert_to_arrow(
        path=output_path,
        time_series=time_series,
        start_times=start_times,
        compression=compression,
    )


if __name__ == "__main__":
    hf_dataset_to_arrow(
        hf_repo="autogluon/chronos_datasets",
        hf_config="training_corpus_tsmixup_10m",
        output_path="/scratch/10608/aadharsh_aadhithya/data/ts/cronos/chronos_training_corpus.arrow",
        freq="D",
        max_series=None,  # set to None to convert all series
        compression="lz4",
        streaming=False,
    )