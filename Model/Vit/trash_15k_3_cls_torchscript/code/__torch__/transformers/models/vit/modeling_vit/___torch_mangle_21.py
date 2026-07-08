class ViTOutput(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  dense : __torch__.torch.nn.modules.linear.___torch_mangle_19.Linear
  dropout : __torch__.torch.nn.modules.dropout.___torch_mangle_20.Dropout
  def forward(self: __torch__.transformers.models.vit.modeling_vit.___torch_mangle_21.ViTOutput,
    argument_1: Tensor,
    input: Tensor) -> Tensor:
    dropout = self.dropout
    dense = self.dense
    _0 = (dropout).forward((dense).forward(argument_1, ), )
    return torch.add(_0, input)
