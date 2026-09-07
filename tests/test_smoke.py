from omegaconf import OmegaConf
import torch

from modeling.losses import ORTARLoss
from modeling.generators import ORTModel


def tiny_config():
    return OmegaConf.create(
        {
            "model": {
                "vq_model": {"codebook_size": 32, "num_latent_tokens": 273},
                "generator": {
                    "hidden_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "dropout": 0.0,
                    "attn_drop": 0.0,
                    "tok_dropout": 0.0,
                    "image_seq_len": 273,
                    "condition_num_classes": 10,
                    "use_checkpoint": False,
                    "randomness_anneal_start": 10,
                    "randomness_anneal_end": 20,
                },
                "loss": {
                    "alpha_weight_random": 0.5,
                    "beta_weight_random": 1.0,
                    "alpha_weight_raster_start": 1.0,
                    "beta_weight_raster_start": 1.0,
                    "alpha_weight_raster_end": 1.0,
                    "beta_weight_raster_end": 1.0,
                    "weight_anneal_start": 0,
                    "weight_anneal_end": 0,
                },
            },
            "training": {"max_train_steps": 30},
        }
    )


def test_forward_loss_and_ordinal_weights():
    torch.manual_seed(0)
    config = tiny_config()
    model = ORTModel(config)
    model.set_random_ratio(1.0)

    tokens = torch.randint(0, 32, (2, 273))
    labels = torch.tensor([1, 2])
    condition = model.preprocess_condition(labels)
    logits, targets, weights = model(tokens, condition, return_labels=True)

    assert logits["x"].shape == (2, 274, 32)
    assert targets.shape == (2, 273)
    assert torch.allclose(weights[:, 0], torch.tensor(0.5).expand(2))
    assert torch.allclose(weights[:, -1], torch.tensor(1.0).expand(2))

    loss, metrics = ORTARLoss(config)(logits, targets, weights)
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["correct_tokens"])
    loss.backward()


def test_randomness_schedule():
    config = tiny_config()
    model = ORTModel(config)
    assert model.get_rar_random_ratio(config, 0) == 1.0
    assert model.get_rar_random_ratio(config, 15) == 0.5
    assert model.get_rar_random_ratio(config, 20) == 0.0


