class Dropout(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.dropout.Dropout,
    input: Tensor) -> Tensor:
    return torch.dropout(input, 0., False)
