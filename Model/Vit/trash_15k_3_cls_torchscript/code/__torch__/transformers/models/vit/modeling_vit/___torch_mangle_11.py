class ViTSelfAttention(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  query : __torch__.torch.nn.modules.linear.___torch_mangle_8.Linear
  key : __torch__.torch.nn.modules.linear.___torch_mangle_9.Linear
  value : __torch__.torch.nn.modules.linear.___torch_mangle_10.Linear
  def forward(self: __torch__.transformers.models.vit.modeling_vit.___torch_mangle_11.ViTSelfAttention,
    argument_1: Tensor) -> Tensor:
    query = self.query
    value = self.value
    key = self.key
    _0 = (key).forward(argument_1, )
    _1 = ops.prim.NumToTensor(torch.size(_0, 0))
    _2 = int(_1)
    _3 = ops.prim.NumToTensor(torch.size(_0, 1))
    x = torch.view(_0, [_2, int(_3), 3, 64])
    key0 = torch.permute(x, [0, 2, 1, 3])
    _4 = (value).forward(argument_1, )
    _5 = ops.prim.NumToTensor(torch.size(_4, 0))
    _6 = int(_5)
    _7 = ops.prim.NumToTensor(torch.size(_4, 1))
    x0 = torch.view(_4, [_6, int(_7), 3, 64])
    value0 = torch.permute(x0, [0, 2, 1, 3])
    _8 = (query).forward(argument_1, )
    _9 = ops.prim.NumToTensor(torch.size(_8, 0))
    _10 = int(_9)
    _11 = ops.prim.NumToTensor(torch.size(_8, 1))
    x1 = torch.view(_8, [_10, int(_11), 3, 64])
    query0 = torch.permute(x1, [0, 2, 1, 3])
    query1 = torch.contiguous(query0)
    key1 = torch.contiguous(key0)
    value1 = torch.contiguous(value0)
    attn_output = torch.scaled_dot_product_attention(query1, key1, value1, None, 0., False, scale=0.125)
    context_layer = torch.contiguous(torch.transpose(attn_output, 1, 2))
    _12 = ops.prim.NumToTensor(torch.size(context_layer, 0))
    _13 = int(_12)
    _14 = ops.prim.NumToTensor(torch.size(context_layer, 1))
    input = torch.reshape(context_layer, [_13, int(_14), 192])
    return input
