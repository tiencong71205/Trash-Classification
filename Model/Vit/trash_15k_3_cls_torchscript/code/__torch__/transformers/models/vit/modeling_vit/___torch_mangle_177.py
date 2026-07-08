class ViTLayer(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  attention : __torch__.transformers.models.vit.modeling_vit.___torch_mangle_168.ViTAttention
  intermediate : __torch__.transformers.models.vit.modeling_vit.___torch_mangle_171.ViTIntermediate
  output : __torch__.transformers.models.vit.modeling_vit.___torch_mangle_174.ViTOutput
  layernorm_before : __torch__.torch.nn.modules.normalization.___torch_mangle_175.LayerNorm
  layernorm_after : __torch__.torch.nn.modules.normalization.___torch_mangle_176.LayerNorm
  def forward(self: __torch__.transformers.models.vit.modeling_vit.___torch_mangle_177.ViTLayer,
    argument_1: Tensor) -> Tensor:
    output = self.output
    intermediate = self.intermediate
    layernorm_after = self.layernorm_after
    attention = self.attention
    layernorm_before = self.layernorm_before
    _0 = (layernorm_before).forward(argument_1, )
    input = torch.add((attention).forward(_0, ), argument_1)
    _1 = (intermediate).forward((layernorm_after).forward(input, ), )
    return (output).forward(_1, input, )
