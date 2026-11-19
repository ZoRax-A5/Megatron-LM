# Layer-wise selective recomputation (experimental)

Use `--recompute-granularity selective --recompute-layer-config '{"0":["attn_qkv"],"13":["mlp_fc1"],"14":["attn_qkv","mlp_fc1"],"27":[]}'`.
Keys are zero-based global Transformer layer numbers, independent of pipeline stage;
an absent key and an empty list both mean no selective recomputation for that layer.
When supplied, this mapping replaces the global `--recompute-modules` selection.
When omitted, existing global behavior is unchanged. The config accepts the same
module names and compatibility constraints as global selective recomputation.
Duplicate keys after integer normalization, duplicate module names, invalid layer
indices and unsupported modules are rejected. Whole-MLP and an internal MLP
checkpoint cannot be combined in the same layer. Different layers may use each.

Layer assignment is made after Megatron resolves its one-based global layer number;
internal MTP layer numbering is not supported. Fine-grained FP8/FP4, sequence
parallel, MLA, graph capture and activation offload remain unsupported. This option
does not implement communication/computation overlap or adaptive scheduling.

For TE grouped MoE experts, `moe_fc1`, `moe_act`, and `moe_fc2` select
individual expert computation boundaries; `moe` selects the whole MoE. For
example, `--recompute-layer-config '{"0":["moe_fc1"],"1":["moe_act"],"2":["moe_fc2"],"3":["moe_fc1","moe_act","moe_fc2"]}'`.
The FC boundaries require bias-free BF16/FP32 TEGroupedMLP, with fused grouped
ops, CUDA graphs, expert communication overlap and activation offload disabled.
Whole MoE and its internal checkpoints cannot be combined within a layer.
MoE selections on dense layers are rejected for layer-wise mappings. Expert
FC selections are rejected for non-TEGroupedMLP implementations. Attention
fine boundaries may be used on MoE models subject to the existing restrictions;
`mlp_fc1`, `mlp_act`, and `mlp_fc2` remain dense-only. Expert parallelism,
other model architectures and attention boundaries on MoE require further validation.
