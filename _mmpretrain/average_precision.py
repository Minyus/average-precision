"""
code from
https://github.com/open-mmlab/mmpretrain/blob/a4c219e05d3ab78c20b9d22dedde7dded6fd206c/mmpretrain/evaluation/metrics/retrieval.py#L373C1-L396C20
"""
from typing import List, Optional

import numpy as np


def calc_apk(target: List, pred: List, mode: str = "IR", k: Optional[int] = None):
    """Added"""
    if not target:
        return 0.0

    if k is None:
        k = len(pred)
    if len(pred) > k:
        pred = pred[:k]

    pred = np.array(pred)
    target = np.array(target)

    num_preds = len(pred)

    positive_ranks = np.arange(num_preds)[np.in1d(pred, target)]

    ap = 0
    for i, rank in enumerate(positive_ranks):
        if mode == "IR":
            precision = (i + 1) / (rank + 1)
            ap += precision
        elif mode == "integrate":
            # code are modified from https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp # noqa:
            old_precision = i / rank if rank > 0 else 1
            cur_precision = (i + 1) / (rank + 1)
            prediction = (old_precision + cur_precision) / 2
            ap += prediction
    ap = ap / len(target)

    return ap
