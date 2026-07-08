class ViTIntermediate(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  dense : __torch__.torch.nn.modules.linear.___torch_mangle_33.Linear
  intermediate_act_fn : __torch__.transformers.activations.___torch_mangle_34.GELUActivation
  def forward(self: __torch__.transformers.models.vit.modeling_vit.___torch_mangle_35.ViTIntermediate,
    argument_1: Tensor) -> Tensor:
    intermediate_act_fn = self.intermediate_act_fn
    dense = self.dense
    _0 = (intermediate_act_fn).forward((dense).forward(argument_1, ), )
    return _0
