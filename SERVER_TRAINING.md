# ActionDiT server training

This guide assumes one RTX 3090, the `ActionDiT` Conda environment, and the
storage layout prepared under `/root/autodl-tmp/actiondit_storage`.

## 1. Activate the environment and caches

```bash
export STORE=/root/autodl-tmp/actiondit_storage
export CONDA_ENVS_PATH="$STORE/envs"
export HF_HOME="$STORE/cache/huggingface"
export TORCH_HOME="$STORE/cache/torch"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ActionDiT
```

PyTorch must be installed separately with the CUDA wheel appropriate for the
server. The remaining dependencies are in `requirements.txt`.

## 2. Define paths

Run all project commands from the `ActionDiT` directory.

```bash
cd "$STORE/repos/myAction-DiT/ActionDiT"

export DATASET_ROOT="$STORE/datasets/libero"
export QWEN_MODEL="$STORE/models/Qwen3-Embedding-0.6B"
export DINO_REPO="$TORCH_HOME/hub/facebookresearch_dinov2_main"
export DINO_WEIGHTS="$TORCH_HOME/hub/checkpoints/dinov2_vits14_pretrain.pth"
```

The dataset root must contain `libero_goal/*.hdf5`. Verify the model files:

```bash
test -d "$QWEN_MODEL"
test -d "$DINO_REPO"
test -f "$DINO_WEIGHTS"
find "$DATASET_ROOT/libero_goal" -maxdepth 1 -name '*.hdf5' | wc -l
```

The final command should print `10`.

## 3. Build the full LIBERO-Goal split and action statistics

These files must be rebuilt because the original checked-in configuration only
contains the `turn_on_the_stove` task.

```bash
python scripts/build_libero_index.py \
  "$DATASET_ROOT" \
  --output configs/libero_goal_split.json \
  --val-ratio 0.2 \
  --seed 42

python scripts/compute_action_stats.py \
  "$DATASET_ROOT" \
  configs/libero_goal_split.json \
  --output configs/libero_goal_action_stats.json
```

Inspect the generated files and run the data tests:

```bash
python -m unittest discover -s tests -v
```

## 4. Run a 100-step smoke test

Do this before paying for a long run. It tests both pretrained encoders, BF16,
data loading, validation, TensorBoard, and checkpoint writing.

```bash
python train.py \
  --dataset_root "$DATASET_ROOT" \
  --manifest_path configs/libero_goal_split.json \
  --stats_path configs/libero_goal_action_stats.json \
  --qwen_model_path "$QWEN_MODEL" \
  --dino_repo "$DINO_REPO" \
  --dino_weights "$DINO_WEIGHTS" \
  --output_dir "$STORE/outputs/libero_goal_smoke" \
  --precision bf16 \
  --batch_size 4 \
  --num_workers 4 \
  --num_epochs 1 \
  --max_steps 100 \
  --log_every 10 \
  --val_every 50 \
  --val_batches 10
```

Each training invocation creates a timestamped subdirectory under
`--output_dir`:

```text
libero_goal_smoke/
└── run_YYYYMMDD_HHMMSS_microseconds/
    ├── run_config.json
    ├── train.log
    ├── tensorboard/
    └── best.pt
```

Training messages are written to both the terminal and that run's `train.log`.
After every validation, `best.pt` is replaced only when validation loss improves,
so each run keeps a single best checkpoint. Resumed runs also create a new run
subdirectory.

## 5. Run one complete pass over LIBERO-Goal

`max_steps=0` means that the epoch count controls termination. Start with one
full epoch to measure runtime and loss behavior:

```bash
python train.py \
  --dataset_root "$DATASET_ROOT" \
  --manifest_path configs/libero_goal_split.json \
  --stats_path configs/libero_goal_action_stats.json \
  --qwen_model_path "$QWEN_MODEL" \
  --dino_repo "$DINO_REPO" \
  --dino_weights "$DINO_WEIGHTS" \
  --output_dir "$STORE/outputs/libero_goal_epoch1" \
  --precision bf16 \
  --batch_size 8 \
  --num_workers 8 \
  --num_epochs 1 \
  --max_steps 0 \
  --log_every 20 \
  --val_every 1000 \
  --val_batches 100
```

If batch size 8 runs out of memory, retry with 4. If memory usage is comfortably
below 24 GB, batch size can be increased for a later experiment.

Run long jobs inside `tmux` and monitor them with `nvidia-smi` and TensorBoard:

```bash
tensorboard --logdir "$STORE/outputs" --host 0.0.0.0 --port 6006
```

## 6. Resume a stopped job

Pass an existing checkpoint and set `num_epochs` to the total desired epoch
count. A resumed mid-epoch checkpoint starts that epoch again, so a small amount
of data can be repeated.

```bash
python train.py \
  --dataset_root "$DATASET_ROOT" \
  --manifest_path configs/libero_goal_split.json \
  --stats_path configs/libero_goal_action_stats.json \
  --qwen_model_path "$QWEN_MODEL" \
  --dino_repo "$DINO_REPO" \
  --dino_weights "$DINO_WEIGHTS" \
  --output_dir "$STORE/outputs/libero_goal_epoch1" \
  --precision bf16 \
  --batch_size 8 \
  --num_workers 8 \
  --num_epochs 1 \
  --resume "$STORE/outputs/libero_goal_epoch1/run_YYYYMMDD_HHMMSS_microseconds/best.pt"
```

Checkpoints contain only trainable ActionDiT parameters and optimizer state.
Frozen Qwen and DINO weights are reloaded from their original local paths.
