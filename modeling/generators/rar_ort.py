"""ORT curriculum on the original RAR architecture, preserving checkpoint keys."""
from modeling.generators.rar_original import RAR
from modeling.generators.ort import ORTModel


class RARORT(RAR):
    # Share the established curriculum, not the AliTok network architecture.
    get_rar_random_ratio = ORTModel.get_rar_random_ratio
    get_rar_alpha_beta_weight = ORTModel.get_rar_alpha_beta_weight
    set_alpha_beta_weight = ORTModel.set_alpha_beta_weight

    def __init__(self, config):
        super().__init__(config)
        self.alpha_weight_random = config.model.loss.alpha_weight_random
        self.beta_weight_random = config.model.loss.beta_weight_random
        self.alpha_weight_raster = config.model.loss.alpha_weight_raster_start
        self.beta_weight_raster = config.model.loss.beta_weight_raster_start

    def forward(self, input_ids, condition, return_labels=False):
        orders, weights = ORTModel.sample_orders(self, input_ids)
        outputs = self.forward_fn(input_ids, condition, return_labels, orders)
        if return_labels:
            logits, labels = outputs
            return {"x": logits, "x_": None}, labels, weights
        return outputs
