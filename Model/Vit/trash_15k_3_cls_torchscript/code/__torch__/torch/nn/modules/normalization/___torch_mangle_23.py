class LayerNorm(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.normalization.___torch_mangle_23.LayerNorm,
    input: Tensor) -> Tensor:
    bias = self.bias
    weight = self.weight
    input0 = torch.layer_norm(input, [192], weight, bias, 9.9999999999999998e-13)
    return input0
