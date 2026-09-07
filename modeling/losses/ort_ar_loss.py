from typing import Mapping, Text, Tuple
import torch
from utils.registry import register_loss

@register_loss("ort_ar_loss")
class ORTARLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.target_vocab_size = config.model.vq_model.codebook_size
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor, *args) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:

        logits, logits_ = logits['x'], logits['x_']

        shift_logits = logits[..., :-1, :].permute(0, 2, 1).contiguous()  # NLC->NCL
        shift_labels = labels.contiguous()
        shift_logits = shift_logits.view(shift_logits.shape[0], self.target_vocab_size, -1)
        shift_labels = shift_labels.view(shift_labels.shape[0], -1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = self.criterion(shift_logits, shift_labels)
        loss *= weight if weight is not None else 1

        # shift_logits_ = logits_[..., :-1, :].permute(0, 2, 1).contiguous()  # NLC->NCL
        # shift_logits_ = shift_logits_.view(shift_logits_.shape[0], self.target_vocab_size, -1)
        # loss_ = self.criterion(shift_logits_, shift_labels)
        # loss_ *= weight if weight is not None else 1

        # loss *= 1/weight.mean()
        correct_tokens = (torch.argmax(shift_logits, dim=1) == shift_labels).sum(dim=1) / shift_labels.size(1)
        return loss.mean(), {"loss_ar": loss.mean(), "correct_tokens": correct_tokens.float().mean()} #"loss_mid_level_ar": loss_.mean()