class Conv2d(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.conv.Conv2d,
    pixel_values: Tensor) -> Tensor:
    bias = self.bias
    weight = self.weight
    _0 = torch._convolution(pixel_values, weight, bias, [16, 16], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
    return _0
