class WrappedModel(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  model : __torch__.transformers.models.vit.modeling_vit.ViTForImageClassification
  def forward(self: __torch__.WrappedModel,
    pixel_values: Tensor) -> Tensor:
    model = self.model
    return (model).forward(pixel_values, )
