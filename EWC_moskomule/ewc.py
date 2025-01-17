from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

class EWC(object):
    def __init__(self, model: nn.Module, dataset: list, device):

        self.model = model
        self.dataset = dataset

        self.device = device # add

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for batch in self.dataset: # input
            self.model.zero_grad()
            # input = variable(input)
            batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
            inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'labels': batch[3]}
            inputs['token_type_ids'] = batch[2]

            output = self.model(**inputs) # .view(1, -1)
            # label = output.max(1)[1].view(-1)
            # loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss = output[0]
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss