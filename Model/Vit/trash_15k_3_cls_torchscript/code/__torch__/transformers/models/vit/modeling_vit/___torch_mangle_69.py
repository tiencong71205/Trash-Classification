class ViTIntermediate(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  dense : __torch__.torch.nn.modules.linear.___torch_mangle_67.Linear
  intermediate_act_fn : __torch__.transformers.activations.___torch_mangle_68.GELUActivation
  def forward(self: __torch__.transformers.models.vit.modeling_vit.___torch_mangle_69.ViTIntermediate,
    argument_1: Tensor) -> Tensor:
    intermediate_act_fn = self.intermediate_act_fn
    dense = self.dense
    _0 = (intermediate_act_fn).forward((dense).forward(argument_1, ), )
    return _0
