class Dropout(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.dropout.___torch_mangle_115.Dropout,
    argument_1: Tensor) -> Tensor:
    attention_output = torch.dropout(argument_1, 0., False)
    return attention_output
