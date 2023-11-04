from typing import Any, List

import numpy as np
from _ml_metrics.average_precision import apk
from _sklearn.average_precision import average_precision_score


def indices(ls: List, element: Any):
    return [i for i in range(len(ls)) if ls[i] == element]


def sort_2_lists(list1: List, list2: List, reverse: bool = False):
    list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2), reverse=reverse)))
    return list1, list2


def apk_multihot(y_true: List, y_scores: List, k: int = 1):
    y_scores, y_true = sort_2_lists(y_scores, y_true, reverse=True)
    gt_indices = indices(y_true, 1)
    return apk(gt_indices, list(range(k)))


if __name__ == "__main__":
    rng = np.random.default_rng()

    for p in [0.2, 0.5, 0.8]:
        y_true = (rng.random(10) < p).astype(int).tolist()

        y_scores = list(range(len(y_true)))[::-1]

        for k in range(1, 13):
            mm_apk = apk_multihot(y_true, y_scores, k=k)
            sk_apk = average_precision_score(y_true, y_scores, k=k)

            assert abs(mm_apk - sk_apk) < 1e-6, f"top{k}: {mm_apk} != {sk_apk}"
            # print(f"[k={k}]", "mm_apk: ", mm_apk, "sk_apk:", sk_apk, "\n")
    print(f"Average Precision matched for y_true = {y_true}")
