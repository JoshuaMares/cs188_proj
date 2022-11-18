import csv
import glob
import json
import random
import logging
import os
from enum import Enum
from typing import List, Optional, Union

import tqdm
import numpy as np

import torch
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

def sort_crit(tup):
    return tup[0]

def pairwise_accuracy(guids, preds, labels):

    acc = 0.0  # The accuracy to return.

    ########################################################
    # TODO: Please finish the pairwise accuracy computation.
    # Hint: Utilize the `guid` as the `guid` for each
    # statement coming from the same complementary
    # pair is identical. You can simply pair the these
    # predictions and labels w.r.t the `guid`.
    list = []
    for i in range(0, len(guids)):
        list.append((guids[i], preds[i], labels[i]))

    list.sort(key = sort_crit)
    for i in range(0, len(list), 2):
        if list[i][1] == list[i][2] and list[i+1][1] == list[i+1][2]:
            acc += 1

    acc = acc*2 / len(list)




    #raise NotImplementedError("Please finish the TODO!")
    # End of TODO
    ########################################################

    return acc


if __name__ == "__main__":

    # Unit-testing the pairwise accuracy function.
    guids = [0, 0, 1, 1, 2, 2, 3, 3]
    preds = np.asarray([0, 0, 1, 0, 0, 1, 1, 1])
    labels = np.asarray([1, 0,1, 0, 0, 1, 1, 1])
    acc = pairwise_accuracy(guids, preds, labels)

    if acc == 0.75:
        print("Your `pairwise_accuracy` function is correct!")
    else:
        raise NotImplementedError("Your `pairwise_accuracy` function is INCORRECT!")
