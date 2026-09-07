from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from modeling.factory import build_model, build_loss, build_tokenizer, configure_order, training_outputs


def tiny_config():
    c = OmegaConf.load(Path(__file__).resolve().parents[1] / 'configs/ort_l_rar_xl_400.yaml')
    c.model.generator.hidden_size = 32
    c.model.generator.num_hidden_layers = 1
    c.model.generator.num_attention_heads = 4
    c.model.generator.intermediate_size = 128
    c.model.generator.dropout = 0
    c.model.generator.attn_drop = 0
    return c


@pytest.mark.parametrize('alpha,beta', [(0.75, 1.25), (1, 0)])
def test_rar_ort_preserves_architecture_and_weights_by_generation_position(alpha, beta):
    c = tiny_config()
    c.model.loss.alpha_weight_random = alpha
    c.model.loss.beta_weight_random = beta
    m = build_model(c)
    original = OmegaConf.create(OmegaConf.to_container(c))
    original.model.generator.type = 'rar_original'
    baseline = build_model(original)
    baseline.load_state_dict(m.state_dict(), strict=True)
    baseline.eval()
    m.eval()
    tokens = torch.arange(256).unsqueeze(0)
    condition = m.preprocess_condition(torch.tensor([1]), cond_drop_prob=0)
    configure_order(m, c, 0)
    logits, labels, weight = training_outputs(m, tokens, condition, c)
    # Unique tokens reveal the actual permutation; logits must match original RAR.
    expected_logits, expected_labels = baseline.forward_fn(tokens, condition, True, labels)
    torch.testing.assert_close(logits['x'], expected_logits)
    torch.testing.assert_close(labels, expected_labels)
    torch.testing.assert_close(weight[0], torch.linspace(alpha, beta, 256))
    loss, _ = build_loss(c)(logits, labels, weight)
    ce = torch.nn.functional.cross_entropy(logits['x'][:, :-1].transpose(1, 2), labels, reduction='none')
    torch.testing.assert_close(loss, (ce * weight).mean())
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    configure_order(m, c, 156250)
    assert m.random_ratio == 0.5
    configure_order(m, c, 187500)
    logits, labels, weight = training_outputs(m, tokens, condition, c)
    torch.testing.assert_close(labels, tokens)
    torch.testing.assert_close(weight, torch.ones_like(weight))
    baseline_logits, _ = baseline.forward_fn(tokens, condition, True)
    torch.testing.assert_close(logits['x'], baseline_logits)
    with torch.inference_mode():
        sampled = m.generate(condition=torch.tensor([1]), guidance_scale=1., guidance_scale_pow=1., randomize_temperature=1.)
    assert sampled.shape == (1, 256)
    assert sampled.min() >= 0 and sampled.max() < 1024


def test_rar_ort_uses_maskgit_tokenizer():
    c = tiny_config()
    with patch('modeling.tokenizers.maskgit.PretrainedTokenizer') as decoder:
        assert build_tokenizer(c, 'example.bin') is decoder.return_value
        decoder.assert_called_once_with('example.bin')
