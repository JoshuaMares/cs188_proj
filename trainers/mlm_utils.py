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

def mask_tokens(inputs, tokenizer, args, special_tokens_mask=None):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK,
    10% random, 10% original.
    inputs should be tokenized token ids with size: (batch size X input length).
    """

    # The eventual labels will have the same size of the inputs,
    # with the masked parts the same as the input ids but the rest as
    # args.mlm_ignore_index, so that the cross entropy loss will ignore it.
    labels = inputs.clone()
    tmp_inputs = inputs.clone()
    # Constructs the special token masks.
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    ##################################################
    # Optional TODO: this is an optional TODO that can get you more familiarized
    # with masked language modeling.
    #use mlm on c2s then finetune on c2s
    #do classification on semeval and then com2sense
        #not pretaining on semval, just doing regular training

    # First sample a few tokens in each sequence for the MLM, with probability
    # `args.mlm_probability`.
    # Hint: you may find these functions handy: `torch.full`, Tensor's built-in
    # function `masked_fill_`, and `torch.bernoulli`.
    # Check the inputs to the bernoulli function and use other hinted functions
    # to construct such inputs.
    #full will give use array full of mlm_probability
    #bernoulli will give 1 or 0 for each of those cells in the array based on
    #bernoulli
    x = torch.bernoulli(torch.full(labels.shape, args.mlm_probability))

    #sample 80% of words and get new masking and use new masking with old mask
    # Remember that the "non-masked" parts should be filled with ignore index.


    # For 80% of the time, we will replace masked input tokens with  the
    # tokenizer.mask_token (e.g. for (De)BERT it is [MASK] for for RoBERTa it is
    # <mask>, check tokenizer documentation for more details)
    indices_replaced = torch.bernoulli(torch.full(labels.shape, .8)).bool() & x.bool()
    #print(indices_replaced)

    # For 10% of the time, we replace masked input tokens with random word.
    # Hint: you may find function `torch.randint` handy.
    # Hint: make sure that the random word replaced positions are not overlapping
    # with those of the masked positions, i.e. "~indices_replaced".
    indices_random = (torch.bernoulli(torch.full(labels.shape, .1)).bool() & x.bool())
    indices_random = (indices_random ^ indices_replaced) & indices_random
    #print(indices_random)

    indices_replaced = indices_replaced.to(tmp_inputs.device)
    indices_random = indices_random.to(tmp_inputs.device)
    tmp_inputs = tmp_inputs.masked_fill_(indices_replaced == True, 103)
    tmp_inputs = tmp_inputs.masked_fill_(indices_random == True, torch.randint(0, tokenizer.vocab_size, ()))
    #print(tmp_inputs)

    labels = labels.masked_fill_(tmp_inputs != 103, -100)
    inputs = tmp_inputs
    #print(labels)
    # End of TODO
    ##################################################

    # For the rest of the time (10% of the time) we will keep the masked input
    # tokens unchanged
    pass  # Do nothing.

    return inputs, labels


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    pass


if __name__ == "__main__":

    class mlm_args(object):
        def __init__(self):
            self.mlm_probability = 0.4
            self.mlm_ignore_index = -100
            self.device = "cpu"
            self.seed = 42
            self.n_gpu = 0

    args = mlm_args()
    set_seed(args)

    # Unit-testing the MLM function.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    input_sentence = "I am a good student and I love NLP."
    input_ids = tokenizer.encode(input_sentence)
    input_ids = torch.Tensor(input_ids).long().unsqueeze(0)
    #print(input_ids)

    inputs, labels = mask_tokens(input_ids, tokenizer, args,
                                 special_tokens_mask=None)
    inputs, labels = list(inputs.numpy()[0]), list(labels.numpy()[0])
    ans_inputs = [101,   146,  103,  170,  103, 2377,  103,  146, 1567,   103, 2101,  119,  102]
    ans_labels = [-100, -100, 1821, -100, 1363, -100, 1105, -100, -100, 21239, -100, -100, -100]

    if inputs == ans_inputs and labels == ans_labels:
        print("Your `mask_tokens` function is correct!")
    else:
        raise NotImplementedError("Your `mask_tokens` function is INCORRECT!")
