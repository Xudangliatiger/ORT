"""Adapters for original baselines; model files and checkpoint keys stay unchanged."""
import torch
from modeling.generators import ORTModel
from modeling.losses import ORTARLoss
from modeling.losses.ar_loss import ARLoss


def build_model(config):
    kind = config.model.generator.type
    if kind == 'ort':
        return ORTModel(config)
    if kind == 'alitok_original':
        from modeling.generators.alitok_original import ARModel
        return ARModel(config)
    if kind == 'rar_original':
        from modeling.generators.rar_original import RAR
        return RAR(config)
    raise ValueError(f'Unsupported generator: {kind}')


def configure_order(model, config, step):
    kind = config.model.generator.type
    if kind == 'ort':
        model.set_random_ratio(model.get_rar_random_ratio(config, step))
        model.set_alpha_beta_weight(*model.get_rar_alpha_beta_weight(config, step))
    elif kind == 'rar_original':
        start = config.model.generator.randomness_anneal_start
        end = config.model.generator.randomness_anneal_end
        ratio = 1.0 if step < start else 0.0 if step >= end else 1.0 - (step-start)/(end-start)
        model.set_random_ratio(ratio)


def training_outputs(model, tokens, condition, config):
    outputs = model(tokens, condition, return_labels=True)
    if config.model.generator.type == 'ort':
        return outputs
    logits, labels = outputs
    return logits, labels, None


class BaselineLoss(ARLoss):
    def forward(self, logits, labels, weight=None):
        return super().forward(logits, labels)


def build_loss(config):
    return ORTARLoss(config) if config.model.generator.type == 'ort' else BaselineLoss(config)


def build_tokenizer(config, weight_path):
    if config.model.generator.type == 'rar_original':
        from modeling.tokenizers.maskgit import PretrainedTokenizer
        return PretrainedTokenizer(weight_path)
    from modeling.tokenizers import AliTok
    decoder = AliTok()
    decoder.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True), strict=True)
    return decoder
