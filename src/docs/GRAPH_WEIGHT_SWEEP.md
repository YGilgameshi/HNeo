# Graph-weight sensitivity experiment

## Question

The experiment tests whether the current tissue contribution of 0.3 is
supported empirically:

```text
w_ij = (1 - tissue_graph_weight) * sim_HLA(i, j)
       + tissue_graph_weight * I_same_tissue(i, j)
```

The graph threshold remains fixed at 0.3. Because changing the mixture also
changes which edges survive this threshold, every run records edge count,
density, connected components and edge-weight statistics in
`task_graph/graph_statistics.json`.

## Recommended design

- Tissue weights: 0.0, 0.1, ..., 1.0.
- Seeds: 42, 43 and 44 for the initial sweep.
- Fixed overlap-free train/validation/test split.
- Fixed negative-sample cache across all runs.
- Select the weight using validation mean-task AUROC only.
- Report test mean-task AUROC, AUPRC and F1 after selection.
- If 0.2--0.4 are close, rerun 0.2, 0.25, 0.3, 0.35 and 0.4 with five seeds.

## Run

For a small sequential run from the project root:

```bash
python scripts/run_graph_weight_sweep.py \
  --presplit_dir output/mode2_overlap8_groupsplit/data_splits \
  --output_root output/graph_weight_sweep \
  --hla_sequences configs/hla_sequences.json \
  --proteome_file data/human_proteome.fasta \
  --n_epochs 50 \
  --batch_size 256
```

For the full HPC experiment, use the supplied Slurm workflow instead. It first
creates one content-verified negative-sample cache, then launches a 33-task job
array (11 weights x 3 seeds), and finally aggregates the completed runs:

```bash
bash slurm/submit_graph_weight_sweep.sh
```

The default project root and paths match the current HPC layout. The array is
capped at four simultaneous GPU jobs (`%4`); adjust this only if the allocation
permits more concurrent GPUs. The summary job currently uses the `gpu4090`
partition for compatibility with the known cluster configuration, although it
does not computationally require a GPU.

The sweep script enables task-balanced sampling and task weighting by default,
matching the documented full-model command. Use `--disable_task_balanced` or
`--disable_task_weighting` only if the reference 0.3 run used different flags.

## Dry run

Use this first to inspect all generated commands without training:

```bash
python scripts/run_graph_weight_sweep.py \
  --presplit_dir /path/to/data_splits \
  --output_root output/graph_weight_sweep \
  --dry_run
```

## Outputs

- `graph_weight_runs.csv`: one row per weight and seed.
- `graph_weight_summary.csv`: mean and standard deviation across seeds.
- `graph_weight_summary.json`: machine-readable results and selected weight.
- `graph_weight_sensitivity.pdf/png`: validation/test sensitivity curve.
- `sweep_manifest.json`: exact command for every run.
- `tw_*/seed_*/run.log`: complete log for each run.

Do not claim that 0.3 is optimal unless it ranks first on validation performance
or is statistically indistinguishable from the best value while offering a
clearer/sparser graph. The test set must not be used to choose the weight.
