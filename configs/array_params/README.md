# Array Param Files

Use these files with OAR arrays so you can launch each campaign separately without editing/commenting lines.

Example:

```bash
oarsub --array-param-file configs/array_params/ablations.txt train.oar
```

Available groups:

- `configs/array_params/main_unimp_20.txt`
- `configs/array_params/main_unimp_50.txt`
- `configs/array_params/main_unimp_100.txt`
- `configs/array_params/main_unimp_all_sizes.txt`
- `configs/array_params/mtvrp_moe_4_experts_50.txt` — MTVRP-50 MoE end-to-end and recourse runs with four experts
- `configs/array_params/main_recourse.txt`
- `configs/array_params/main_recourse_models_20_50_100.txt`
- `configs/array_params/ablations.txt`
- `configs/array_params/knn.txt`
- `configs/array_params/pdptw_scratch.txt` — PDPTW recourse from scratch at sizes 20/50/100
- `configs/array_params/pdptw_transfer.txt` — PDPTW recourse with weights transferred from MTVRP recourse checkpoints (requires `MTVRP_CKPT_{20,50,100}` env vars)
- `configs/array_params/pdptw_adapter_transfer_20_50.txt` — PDPTW adapter transfer for sizes 20/50. Uses `lr=5e-5` so the small pickup-delivery adapters learn without aggressively changing transferred time-window/capacity behavior. The LR still follows the normal `MultiStepLR`: with default milestones `[270, 295]`, `5e-5 -> 5e-6 -> 5e-7`.
- `configs/array_params/pdptw_all.txt` — both campaigns combined
- `configs/array_params/aco_decomposition_500.txt` — 100-epoch CVRP-500 ACO decomposition training
