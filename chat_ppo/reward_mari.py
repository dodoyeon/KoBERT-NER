import torch 

class reward_algorithm():
    def __init__(self) -> None:
        pass
    def forward(outputs, labels, attention_mask): # prev_pred
        # preds = torch.argmax(outputs, dim=2)
        # compare = torch.eq(labels, preds)
        # nsum = torch.sum(attention_mask)
        # return (-5*torch.sum(compare))/nsum
        pass