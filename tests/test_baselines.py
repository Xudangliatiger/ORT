from pathlib import Path
import pytest
import torch
from omegaconf import OmegaConf
from modeling.factory import build_model, build_loss, configure_order, training_outputs

@pytest.mark.parametrize('name', ['alitok_xl_original_400', 'rar_xl_original_400'])
def test_original_baseline_forward_backward_and_sample(name):
    c=OmegaConf.load(Path(__file__).resolve().parents[1]/'configs'/f'{name}.yaml')
    c.model.generator.hidden_size=32
    c.model.generator.num_hidden_layers=1
    c.model.generator.num_attention_heads=4
    c.model.generator.intermediate_size=128
    c.model.vq_model.codebook_size=32
    c.model.generator.condition_num_classes=10
    m=build_model(c)
    configure_order(m,c,0)
    tokens=torch.randint(0,32,(1,c.model.generator.image_seq_len))
    labels=torch.tensor([1])
    loss,_=build_loss(c)(*training_outputs(m,tokens,m.preprocess_condition(labels),c))
    assert torch.isfinite(loss)
    loss.backward()
    m.eval()
    with torch.inference_mode():
        result=m.generate(condition=labels,guidance_scale=1.0,guidance_scale_pow=1.0,randomize_temperature=1.0)
    assert result.shape==tokens.shape
    assert result.min()>=0 and result.max()<32
