class ViTForImageClassification(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  vit : __torch__.transformers.models.vit.modeling_vit.ViTModel
  classifier : __torch__.torch.nn.modules.linear.___torch_mangle_196.Linear
  def forward(self: __torch__.transformers.models.vit.modeling_vit.ViTForImageClassification,
    pixel_values: Tensor) -> Tensor:
    classifier = self.classifier
    vit = self.vit
    _0 = torch.slice((vit).forward(pixel_values, ), 0, 0, 9223372036854775807)
    input = torch.slice(torch.select(_0, 1, 0), 1, 0, 9223372036854775807)
    return (classifier).forward(input, )
class ViTModel(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  embeddings : __torch__.transformers.models.vit.modeling_vit.ViTEmbeddings
  encoder : __torch__.transformers.models.vit.modeling_vit.ViTEncoder
  layernorm : __torch__.torch.nn.modules.normalization.___torch_mangle_195.LayerNorm
  def forward(self: __torch__.transformers.models.vit.modeling_vit.ViTModel,
    pixel_values: Tensor) -> Tensor:
    layernorm = self.layernorm
    encoder = self.encoder
    embeddings = self.embeddings
    _1 = (embeddings).forward(pixel_values, )
    _2 = (layernorm).forward((encoder).forward(_1, ), )
    return _2
class ViTEmbeddings(Module):
  __parameters__ = ["cls_token", "position_embeddings", ]
  __buffers__ = []
  cls_token : Tensor
  position_embeddings : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  patch_embeddings : __torch__.transformers.models.vit.modeling_vit.ViTPatchEmbeddings
  dropout : __torch__.torch.nn.modules.dropout.Dropout
  def forward(self: __torch__.transformers.models.vit.modeling_vit.ViTEmbeddings,
    pixel_values: Tensor) -> Tensor:
    dropout = self.dropout
    position_embeddings = self.position_embeddings
    cls_token = self.cls_token
    patch_embeddings = self.patch_embeddings
    batch_size = ops.prim.NumToTensor(torch.size(pixel_values, 0))
    _3 = int(batch_size)
    _4 = (patch_embeddings).forward(pixel_values, )
    cls_tokens = torch.expand(cls_token, [_3, -1, -1])
    embeddings = torch.cat([cls_tokens, _4], 1)
    input = torch.add(embeddings, position_embeddings)
    return (dropout).forward(input, )
class ViTPatchEmbeddings(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  projection : __torch__.torch.nn.modules.conv.Conv2d
  def forward(self: __torch__.transformers.models.vit.modeling_vit.ViTPatchEmbeddings,
    pixel_values: Tensor) -> Tensor:
    projection = self.projection
    _5 = (projection).forward(pixel_values, )
    embeddings = torch.transpose(torch.flatten(_5, 2), 1, 2)
    return embeddings
class ViTEncoder(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  layer : __torch__.torch.nn.modules.container.ModuleList
  def forward(self: __torch__.transformers.models.vit.modeling_vit.ViTEncoder,
    argument_1: Tensor) -> Tensor:
    layer = self.layer
    _11 = getattr(layer, "11")
    layer0 = self.layer
    _10 = getattr(layer0, "10")
    layer1 = self.layer
    _9 = getattr(layer1, "9")
    layer2 = self.layer
    _8 = getattr(layer2, "8")
    layer3 = self.layer
    _7 = getattr(layer3, "7")
    layer4 = self.layer
    _6 = getattr(layer4, "6")
    layer5 = self.layer
    _5 = getattr(layer5, "5")
    layer6 = self.layer
    _4 = getattr(layer6, "4")
    layer7 = self.layer
    _3 = getattr(layer7, "3")
    layer8 = self.layer
    _2 = getattr(layer8, "2")
    layer9 = self.layer
    _1 = getattr(layer9, "1")
    layer10 = self.layer
    _0 = getattr(layer10, "0")
    _12 = (_1).forward((_0).forward(argument_1, ), )
    _13 = (_4).forward((_3).forward((_2).forward(_12, ), ), )
    _14 = (_7).forward((_6).forward((_5).forward(_13, ), ), )
    _15 = (_10).forward((_9).forward((_8).forward(_14, ), ), )
    return (_11).forward(_15, )
class ViTLayer(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  attention : __torch__.transformers.models.vit.modeling_vit.ViTAttention
  intermediate : __torch__.transformers.models.vit.modeling_vit.ViTIntermediate
  output : __torch__.transformers.models.vit.modeling_vit.ViTOutput
  layernorm_before : __torch__.torch.nn.modules.normalization.LayerNorm
  layernorm_after : __torch__.torch.nn.modules.normalization.___torch_mangle_7.LayerNorm
  def forward(self: __torch__.transformers.models.vit.modeling_vit.ViTLayer,
    argument_1: Tensor) -> Tensor:
    output = self.output
    intermediate = self.intermediate
    layernorm_after = self.layernorm_after
    attention = self.attention
    layernorm_before = self.layernorm_before
    _10 = (layernorm_before).forward(argument_1, )
    input = torch.add((attention).forward(_10, ), argument_1)
    _11 = (intermediate).forward((layernorm_after).forward(input, ), )
    return (output).forward(_11, input, )
class ViTAttention(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  attention : __torch__.transformers.models.vit.modeling_vit.ViTSelfAttention
  output : __torch__.transformers.models.vit.modeling_vit.ViTSelfOutput
  def forward(self: __torch__.transformers.models.vit.modeling_vit.ViTAttention,
    argument_1: Tensor) -> Tensor:
    output = self.output
    attention = self.attention
    _12 = (output).forward((attention).forward(argument_1, ), )
    return _12
class ViTSelfAttention(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  query : __torch__.torch.nn.modules.linear.Linear
  key : __torch__.torch.nn.modules.linear.___torch_mangle_0.Linear
  value : __torch__.torch.nn.modules.linear.___torch_mangle_1.Linear
  def forward(self: __torch__.transformers.models.vit.modeling_vit.ViTSelfAttention,
    argument_1: Tensor) -> Tensor:
    query = self.query
    value = self.value
    key = self.key
    _13 = (key).forward(argument_1, )
    _14 = ops.prim.NumToTensor(torch.size(_13, 0))
    _15 = int(_14)
    _16 = ops.prim.NumToTensor(torch.size(_13, 1))
    x = torch.view(_13, [_15, int(_16), 3, 64])
    key0 = torch.permute(x, [0, 2, 1, 3])
    _17 = (value).forward(argument_1, )
    _18 = ops.prim.NumToTensor(torch.size(_17, 0))
    _19 = int(_18)
    _20 = ops.prim.NumToTensor(torch.size(_17, 1))
    x0 = torch.view(_17, [_19, int(_20), 3, 64])
    value0 = torch.permute(x0, [0, 2, 1, 3])
    _21 = (query).forward(argument_1, )
    _22 = ops.prim.NumToTensor(torch.size(_21, 0))
    _23 = int(_22)
    _24 = ops.prim.NumToTensor(torch.size(_21, 1))
    x1 = torch.view(_21, [_23, int(_24), 3, 64])
    query0 = torch.permute(x1, [0, 2, 1, 3])
    query1 = torch.contiguous(query0)
    key1 = torch.contiguous(key0)
    value1 = torch.contiguous(value0)
    attn_output = torch.scaled_dot_product_attention(query1, key1, value1, None, 0., False, scale=0.125)
    context_layer = torch.contiguous(torch.transpose(attn_output, 1, 2))
    _25 = ops.prim.NumToTensor(torch.size(context_layer, 0))
    _26 = int(_25)
    _27 = ops.prim.NumToTensor(torch.size(context_layer, 1))
    input = torch.reshape(context_layer, [_26, int(_27), 192])
    return input
class ViTSelfOutput(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  dense : __torch__.torch.nn.modules.linear.___torch_mangle_2.Linear
  dropout : __torch__.torch.nn.modules.dropout.___torch_mangle_3.Dropout
  def forward(self: __torch__.transformers.models.vit.modeling_vit.ViTSelfOutput,
    argument_1: Tensor) -> Tensor:
    dropout = self.dropout
    dense = self.dense
    _28 = (dropout).forward((dense).forward(argument_1, ), )
    return _28
class ViTIntermediate(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  dense : __torch__.torch.nn.modules.linear.___torch_mangle_4.Linear
  intermediate_act_fn : __torch__.transformers.activations.GELUActivation
  def forward(self: __torch__.transformers.models.vit.modeling_vit.ViTIntermediate,
    argument_1: Tensor) -> Tensor:
    intermediate_act_fn = self.intermediate_act_fn
    dense = self.dense
    _29 = (intermediate_act_fn).forward((dense).forward(argument_1, ), )
    return _29
class ViTOutput(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  dense : __torch__.torch.nn.modules.linear.___torch_mangle_5.Linear
  dropout : __torch__.torch.nn.modules.dropout.___torch_mangle_6.Dropout
  def forward(self: __torch__.transformers.models.vit.modeling_vit.ViTOutput,
    argument_1: Tensor,
    input: Tensor) -> Tensor:
    dropout = self.dropout
    dense = self.dense
    _30 = (dropout).forward((dense).forward(argument_1, ), )
    return torch.add(_30, input)
