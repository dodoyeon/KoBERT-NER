import os
import random
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from transformers import (
    BertConfig,
    DistilBertConfig,
    ElectraConfig,
    ElectraTokenizer,
    BertTokenizer,
    BertForTokenClassification,
    DistilBertForTokenClassification,
    ElectraForTokenClassification
)
from tokenization_kobert import KoBertTokenizer

MODEL_CLASSES = {
    'kobert': (BertConfig, BertForTokenClassification, KoBertTokenizer),
    'distilkobert': (DistilBertConfig, DistilBertForTokenClassification, KoBertTokenizer),
    'bert': (BertConfig, BertForTokenClassification, BertTokenizer),
    'kobert-lm': (BertConfig, BertForTokenClassification, KoBertTokenizer),
    'koelectra-base': (ElectraConfig, ElectraForTokenClassification, ElectraTokenizer),
    'koelectra-small': (ElectraConfig, ElectraForTokenClassification, ElectraTokenizer),
}

MODEL_PATH_MAP = {
    'kobert': 'monologg/kobert',
    'distilkobert': 'monologg/distilkobert',
    'bert': 'bert-base-multilingual-cased',
    'kobert-lm': 'monologg/kobert-lm',
    'koelectra-base': 'monologg/koelectra-base-v3-discriminator',
    'koelectra-small': 'monologg/koelectra-small-discriminator',
}


def get_test_texts(args):
    texts = []
    with open(os.path.join(args.data_dir, args.test_file), 'r', encoding='utf-8') as f:
        for line in f:
            text, _ = line.split('\t')
            text = text.split()
            texts.append(text)

    return texts


def get_labels(args):
    return [label.strip() for label in open(os.path.join('data', 'label.txt'), 'r', encoding='utf-8')]


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True # add
    # torch.backends.cudnn.benchmark = False # add
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    return f1_pre_rec(labels, preds)


def f1_pre_rec(labels, preds):
    return {
        "precision": precision_score(labels, preds, suffix=True),
        "recall": recall_score(labels, preds, suffix=True),
        "f1": f1_score(labels, preds, suffix=True)
    }


def show_report(labels, preds):
    return classification_report(labels, preds, suffix=True)

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    tensor = tensor * mask
    tensor = tensor.sum(dim=dim)
    mask_sum = mask.sum(dim=dim)
    mean = tensor / (mask_sum + 1e-8)
    return mean

def compute_approx_kl(log_probs: torch.Tensor,
                      log_probs_base: torch.Tensor,
                      action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs - log_probs_base
    approx_kl = (log_ratio.exp() - 1) - log_ratio
    if action_mask is not None:
        approx_kl = masked_mean(approx_kl, action_mask, dim=1)
        return approx_kl
    approx_kl = approx_kl.mean(dim=1) 
    


def custom_loss(beta, outputs, base_outputs, mask):
    log_probs = F.log_softmax(outputs[1], dim =-1)
    base_log_probs = F.log_softmax(base_outputs, dim =-1)
    # kl = compute_approx_kl(log_probs, base_log_probs, action_mask=mask)
    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    kl = kl_loss(log_probs, base_log_probs)

    loss = outputs[0]
    return (loss + (beta * kl))
