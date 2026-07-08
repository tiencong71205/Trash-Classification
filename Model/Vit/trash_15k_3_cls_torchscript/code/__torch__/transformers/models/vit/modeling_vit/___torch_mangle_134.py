class ViTAttention(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  attention : __torch__.transformers.models.vit.modeling_vit.___torch_mangle_130.ViTSelfAttention
  output : __torch__.transformers.models.vit.modeling_vit.___torch_mangle_133.ViTSelfOutput
  def forward(self: __torch__.transformers.models.vit.modeling_vit.___torch_mangle_134.ViTAttention,
    argument_1: Tensor) -> Tensor:
    output = self.output
    attention = self.attention
    _0 = (output).forward((attention).forward(argument_1, ), )
    return _0
