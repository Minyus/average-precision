from typing import Any, List

import numpy as np
from _ml_metrics.average_precision import apk
from _mmpretrain.average_precision import calc_apk
from _sklearn.average_precision import average_precision_score


def indices(ls: List, element: Any):
    return [i for i in range(len(ls)) if ls[i] == element]


def sort_2_lists(list1: List, list2: List, reverse: bool = False):
    list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2), reverse=reverse)))
    return list1, list2


def apk_multihot(y_true: List, y_scores: List, k: int = 1):
    y_scores, y_true = sort_2_lists(y_scores, y_true, reverse=True)
    gt_indices = indices(y_true, 1)
    return apk(gt_indices, list(range(k)), k=k)


def calc_apk_multihot(y_true: List, y_scores: List, k: int = 1, mode: str = "IR"):
    y_scores, y_true = sort_2_lists(y_scores, y_true, reverse=True)
    gt_indices = indices(y_true, 1)
    return calc_apk(gt_indices, list(range(k)), k=k, mode=mode)


if __name__ == "__main__":
    rng = np.random.default_rng()

    assert_ml = True
    assert_mp = True

    for p in [0.2, 0.5, 0.8]:
        y_true = (rng.random(10) < p).astype(int).tolist()
        # y_true = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]

        y_scores = list(range(len(y_true)))[::-1]

        for k in range(1, 13):
            sk_apk = average_precision_score(y_true, y_scores, k=k)

            ml_apk = apk_multihot(y_true, y_scores, k=k)
            mp_apk = calc_apk_multihot(y_true, y_scores, mode="IR", k=k)

            print(f"[k={k}] sk={sk_apk}, ml={ml_apk}, mp={mp_apk}\n")

            if assert_ml:
                assert (
                    abs(ml_apk - sk_apk) < 1e-6
                ), f"k={k}: {ml_apk} != {sk_apk} for {y_true}"
            if assert_mp:
                assert (
                    abs(mp_apk - sk_apk) < 1e-6
                ), f"k={k}: {mp_apk} != {sk_apk} for {y_true}"
    print(f"Average Precision compared for y_true = {y_true}")
