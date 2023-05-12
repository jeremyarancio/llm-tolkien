
from torch import float32
from torch import nn


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): 
        return super().forward(x).to(float32)


def change_lm_head(model: nn.Module) -> None:
    model.lm_head = CastOutputToFloat(model.lm_head)


def freeze_params(model: nn.Module) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(float32)
    return model


def prepare_model(model: nn.Module) -> nn.Module:
    model = freeze_params(model)
    model = change_lm_head(model)
    model.gradient_checkpointing_enable() # Reduce number of stored activations
    model.enable_input_requires_grad()
    return model



def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"


# def create_config(**kwargs):
#     for key, value in kwargs.items():
#         setattr(config, key, value)
#     return config