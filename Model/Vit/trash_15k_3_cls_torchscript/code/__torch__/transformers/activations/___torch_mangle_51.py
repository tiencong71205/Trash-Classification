class GELUActivation(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.transformers.activations.___torch_mangle_51.GELUActivation,
    argument_1: Tensor) -> Tensor:
    return torch.gelu(argument_1)
