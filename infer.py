from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

# Suitable defaults for headless GPU servers. Existing user settings win.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from actiondit.action_dit import ActionDiT
from data.action_dataset import ActionDataset
from data.action_normalizer import ActionNormalizer
from diffusion import create_diffusion


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_STORAGE_ROOT = Path("/root/autodl-tmp/actiondit_storage")
RUNTIME_STORAGE_ROOT = Path(
    os.environ.get("STORE", str(DEFAULT_STORAGE_ROOT))
).expanduser()

ACTION_DIM = 7
ACTION_CHUNK = 16
STATE_DIM = 8
TIME_DIM = 128
HIDDEN_DIM = 256
LEARN_SIGMA = True


def default_dino_repo() -> Path:
    configured = os.environ.get("ACTIONDIT_DINO_REPO") or os.environ.get(
        "DINO_REPO"
    )
    if configured:
        return Path(configured).expanduser()
    return (
        RUNTIME_STORAGE_ROOT
        / "cache"
        / "torch"
        / "hub"
        / "facebookresearch_dinov2_main"
    )


def default_dino_weights() -> Path:
    configured = os.environ.get("ACTIONDIT_DINO_WEIGHTS") or os.environ.get(
        "DINO_WEIGHTS"
    )
    if configured:
        return Path(configured).expanduser()
    return (
        RUNTIME_STORAGE_ROOT
        / "cache"
        / "torch"
        / "hub"
        / "checkpoints"
        / "dinov2_vits14_pretrain.pth"
    )


def default_qwen_model_path() -> Path:
    configured = os.environ.get("ACTIONDIT_QWEN_MODEL") or os.environ.get(
        "QWEN_MODEL"
    )
    if configured:
        return Path(configured).expanduser()
    return RUNTIME_STORAGE_ROOT / "models" / "Qwen3-Embedding-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load ActionDiT and run one DDIM smoke-test inference."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a training checkpoint such as outputs/run_xxx/best.pt.",
    )
    parser.add_argument(
        "--stats_path",
        type=Path,
        default=PROJECT_ROOT / "configs" / "libero_goal_action_stats.json",
        help="The same action statistics file that was used for training.",
    )
    parser.add_argument("--dino_repo", type=Path, default=default_dino_repo())
    parser.add_argument(
        "--dino_weights", type=Path, default=default_dino_weights()
    )
    parser.add_argument(
        "--qwen_model_path", type=Path, default=default_qwen_model_path()
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="Number of DDIM sampling steps; start with 50.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="DDIM stochasticity. eta=0 is deterministic for fixed noise.",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16", "fp16"),
        default="bf16",
    )
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument(
        "--language",
        default="put the bowl on the plate",
        help="Dummy instruction used by this first smoke test.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Use zero-valued dummy inputs instead of starting LIBERO.",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=0,
        help="LIBERO Goal task id in [0, 9]; use -1 to evaluate all tasks.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of official initial states evaluated for each task.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=300,
        help="Maximum number of environment actions in one episode.",
    )
    parser.add_argument(
        "--wait_steps",
        type=int,
        default=10,
        help="Zero-action steps after reset, allowing objects to settle.",
    )
    parser.add_argument(
        "--execute_horizon",
        type=int,
        default=1,
        help="Actions executed from each predicted 16-action chunk.",
    )
    parser.add_argument(
        "--render_gpu_device_id",
        type=int,
        default=0,
        help="GPU id passed to LIBERO's off-screen renderer.",
    )
    parser.add_argument(
        "--results_path",
        type=Path,
        help="Optional path at which to save evaluation results as JSON.",
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=None,
        help="Optional directory for one MP4 rollout video per episode.",
    )
    args = parser.parse_args()

    if args.ddim_steps <= 0:
        parser.error("--ddim_steps must be positive")
    if not 0 <= args.eta <= 1:
        parser.error("--eta must be in [0, 1]")
    if not -1 <= args.task_id <= 9:
        parser.error("--task_id must be in [0, 9], or -1 for every task")
    if args.num_episodes <= 0 or args.max_steps <= 0:
        parser.error("--num_episodes and --max_steps must be positive")
    if args.wait_steps < 0:
        parser.error("--wait_steps cannot be negative")
    if not 1 <= args.execute_horizon <= ACTION_CHUNK:
        parser.error(f"--execute_horizon must be in [1, {ACTION_CHUNK}]")
    if args.render_gpu_device_id < 0:
        parser.error("--render_gpu_device_id cannot be negative")
    return args


def precision_dtype(precision: str) -> torch.dtype | None:
    return {
        "fp32": None,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[precision]


def require_file(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} does not exist: {resolved}")
    return resolved


def require_directory(path: Path, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"{name} does not exist: {resolved}")
    return resolved


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    *,
    device: torch.device,
    dino_repo: Path,
    dino_weights: Path,
    qwen_model_path: Path,
    amp_dtype: torch.dtype | None,
) -> ActionDiT:
    # Qwen was loaded in fp32 for fp32 training, otherwise in the AMP dtype.
    qwen_dtype = torch.float32 if amp_dtype is None else amp_dtype
    return ActionDiT(
        action_dim=ACTION_DIM,
        time_dim=TIME_DIM,
        state_dim=STATE_DIM,
        hidden_dim=HIDDEN_DIM,
        depth=6,
        dino_dim=384,
        qwen_dim=1024,
        action_chunk=ACTION_CHUNK,
        learn_sigma=LEARN_SIGMA,
        dino_repo=str(dino_repo),
        dino_weights=str(dino_weights),
        qwen_model_path=str(qwen_model_path),
        qwen_dtype=qwen_dtype,
    ).to(device)


def load_model_weights(model: ActionDiT, checkpoint_path: Path) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint has no 'model' entry: {checkpoint_path}")

    ret = model.load_state_dict(checkpoint["model"], strict=False)
    if ret.unexpected_keys:
        raise KeyError(f"Unexpected keys in checkpoint: {ret.unexpected_keys}")

    allowed_missing_prefixes = (
        "o_embedder.dino.dino.",
        "l_embedder.qwen.qwen.",
    )
    unexpected_missing = [
        key
        for key in ret.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if unexpected_missing:
        raise KeyError(
            f"Unexpected missing model keys: {unexpected_missing[:10]}"
        )

    print(
        f"loaded model weights; {len(ret.missing_keys)} frozen DINO/Qwen "
        "keys came from their pretrained models"
    )
    allowed_missing_prefixes = (
    "o_embedder.dino.dino.",
    "l_embedder.qwen.qwen.",
    )

    unexpected_missing = [
    key
    for key in ret.missing_keys
    if not key.startswith(allowed_missing_prefixes)
    ]

    if unexpected_missing:
        raise KeyError(
        "Unexpected missing model keys: "
        f"{unexpected_missing[:10]}"
        )



def run_ddim_denoising_loop(
    *,
    model: ActionDiT,
    diffusion,
    x: torch.Tensor,
    condition: torch.Tensor,
    action_mask: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    batch_size = x.shape[0]
    device = x.device

    for i in reversed(range(diffusion.num_timesteps)):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        model_kwargs = {"condition": condition, "action_mask": action_mask}
        out = diffusion.ddim_sample(
            model=model,
            x=x,
            t=t,
            clip_denoised=True,
            eta=eta,
            model_kwargs=model_kwargs,
        )
        x = out["sample"]
    return x




def validate_inputs(
    states: torch.Tensor,
    observations: torch.Tensor,
    language: list[str],
) -> None:
    batch_size = states.shape[0]
    expected_state_shape = (batch_size, STATE_DIM)
    expected_observation_shape = (batch_size, 3, 224, 224)
    if tuple(states.shape) != expected_state_shape:
        raise ValueError(
            f"Expected states {expected_state_shape}, got {tuple(states.shape)}"
        )
    if tuple(observations.shape) != expected_observation_shape:
        raise ValueError(
            "Expected observations "
            f"{expected_observation_shape}, got {tuple(observations.shape)}"
        )
    if len(language) != batch_size:
        raise ValueError(
            f"Expected {batch_size} language instructions, got {len(language)}"
        )
    if states.device != observations.device:
        raise ValueError("states and observations must be on the same device")
    if not torch.isfinite(states).all() or not torch.isfinite(observations).all():
        raise ValueError("Inference inputs contain NaN or Inf")


@torch.inference_mode()
def infer_action_chunk(
    *,
    model: ActionDiT,
    diffusion,
    normalizer: ActionNormalizer,
    states: torch.Tensor,
    observations: torch.Tensor,
    language: list[str],
    eta: float,
    amp_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_inputs(states, observations, language)
    batch_size = states.shape[0]
    device = states.device

    action_mask = torch.ones(
        batch_size,
        ACTION_CHUNK,
        dtype=torch.bool,
        device=device,
    )
    x = torch.randn(
        batch_size,
        ACTION_CHUNK,
        ACTION_DIM,
        device=device,
        dtype=states.dtype,
    )

    use_autocast = device.type == "cuda" and amp_dtype is not None
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=use_autocast,
    ):
        condition = model.encode_condition(states, observations, language)
        normalized_actions = run_ddim_denoising_loop(
            model=model,
            diffusion=diffusion,
            x=x,
            condition=condition,
            action_mask=action_mask,
            eta=eta,
        )

    normalized_actions = normalized_actions.float().clamp(-1, 1)
    raw_actions = normalizer.denormalize(normalized_actions)
    return normalized_actions, raw_actions


def libero_observation_to_model_inputs(
    obs: dict,
    instruction: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Convert one raw LIBERO observation into ActionDiT inputs."""
    try:
        from robosuite.utils import transform_utils as transform_utils
    except ImportError as exc:
        raise ImportError(
            "robosuite/LIBERO is unavailable; run inference in the server "
            "environment where LIBERO is installed"
        ) from exc

    required_keys = (
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "agentview_image",
    )
    missing_keys = [key for key in required_keys if key not in obs]
    if missing_keys:
        raise KeyError(f"LIBERO observation is missing keys: {missing_keys}")

    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    axis_angle = np.asarray(
        transform_utils.quat2axisangle(eef_quat.copy()),
        dtype=np.float32,
    )
    state_array = np.concatenate((eef_pos, axis_angle, gripper), axis=0)
    if state_array.shape != (STATE_DIM,):
        raise ValueError(
            f"Expected an {STATE_DIM}-D LIBERO state, got {state_array.shape}"
        )

    # LIBERO's dataset creator stores obs["agentview_image"] directly as
    # agentview_rgb.  Therefore no flip or rotation is applied here.
    image = np.asarray(obs["agentview_image"], dtype=np.uint8)
    image_tensor = ActionDataset._process_image(image)

    states = torch.from_numpy(state_array).unsqueeze(0).to(device)
    observations = image_tensor.unsqueeze(0).to(device)
    return states, observations, [instruction]


def create_libero_goal_suite():
    try:
        from libero.libero import benchmark
    except ImportError as exc:
        raise ImportError(
            "LIBERO is not installed in the active Python environment"
        ) from exc

    benchmark_dict = benchmark.get_benchmark_dict()
    if "libero_goal" not in benchmark_dict:
        raise KeyError("The installed LIBERO package has no libero_goal suite")
    return benchmark_dict["libero_goal"]()


def load_libero_init_states(task) -> torch.Tensor:
    """Load trusted official LIBERO init states with PyTorch 2.6+."""
    from libero.libero import get_libero_path

    init_states_path = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    )
    if not init_states_path.is_file():
        raise FileNotFoundError(
            f"LIBERO init states do not exist: {init_states_path}"
        )

    # LIBERO's files contain NumPy objects and predate the PyTorch 2.6 change
    # that made weights_only=True the default.  These files come from the
    # trusted official LIBERO repository installed by the user.
    init_states = torch.load(
        init_states_path,
        map_location="cpu",
        weights_only=False,
    )
    return init_states


def evaluate_libero_task(
    *,
    task_suite,
    task_id: int,
    model: ActionDiT,
    diffusion,
    normalizer: ActionNormalizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    eta: float,
    num_episodes: int,
    max_steps: int,
    wait_steps: int,
    execute_horizon: int,
    seed: int,
    render_gpu_device_id: int,
    video_dir: Path | None,
) -> dict:
    """Evaluate one LIBERO Goal task on official fixed initial states."""
    try:
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as exc:
        raise ImportError(
            "LIBERO environments are unavailable in this Python environment"
        ) from exc

    task = task_suite.get_task(task_id)
    instruction = task.language
    init_states = load_libero_init_states(task)
    if num_episodes > len(init_states):
        raise ValueError(
            f"Task {task_id} has only {len(init_states)} official initial "
            f"states, but {num_episodes} episodes were requested"
        )

    env = OffScreenRenderEnv(
        bddl_file_name=task_suite.get_task_bddl_file_path(task_id),
        camera_heights=128,
        camera_widths=128,
        render_gpu_device_id=render_gpu_device_id,
    )
    env.seed(seed)

    cv2 = None
    if video_dir is not None:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "Video recording requires OpenCV (cv2)"
            ) from exc
        video_dir = video_dir.expanduser().resolve()
        video_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    episode_steps: list[int] = []
    video_paths: list[str] = []
    print(f"\ntask {task_id}: {instruction}")

    video_writer = None
    try:
        for episode_id in range(num_episodes):
            env.reset()
            obs = env.set_init_state(init_states[episode_id])

            # Let objects settle before asking the policy for an action.
            zero_action = np.zeros(ACTION_DIM, dtype=np.float32)
            for _ in range(wait_steps):
                obs, _, _, _ = env.step(zero_action)

            video_path = None
            if video_dir is not None:
                video_path = (
                    video_dir
                    / f"task_{task_id:02d}_episode_{episode_id + 1:03d}.mp4"
                )
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    str(video_path), fourcc, 20.0, (128, 128)
                )
                if not video_writer.isOpened():
                    raise RuntimeError(f"Could not create video: {video_path}")
                frame = np.asarray(obs["agentview_image"], dtype=np.uint8)
                video_writer.write(np.ascontiguousarray(frame[..., ::-1]))

            success = bool(env.check_success())
            steps = 0
            while steps < max_steps and not success:
                states, observations, language = (
                    libero_observation_to_model_inputs(
                        obs,
                        instruction,
                        device,
                    )
                )
                _, raw_actions = infer_action_chunk(
                    model=model,
                    diffusion=diffusion,
                    normalizer=normalizer,
                    states=states,
                    observations=observations,
                    language=language,
                    eta=eta,
                    amp_dtype=amp_dtype,
                )

                action_chunk = raw_actions[0, :execute_horizon].cpu().numpy()
                for action in action_chunk:
                    obs, _, done, _ = env.step(action.tolist())
                    steps += 1
                    if video_writer is not None:
                        frame = np.asarray(
                            obs["agentview_image"], dtype=np.uint8
                        )
                        video_writer.write(
                            np.ascontiguousarray(frame[..., ::-1])
                        )
                    success = bool(done) or bool(env.check_success())
                    if success or steps >= max_steps:
                        break

            if video_writer is not None:
                video_writer.release()
                video_writer = None
                video_paths.append(str(video_path))
                print(f"  saved video: {video_path}")

            successes += int(success)
            episode_steps.append(steps)
            running_rate = successes / (episode_id + 1)
            print(
                f"  episode {episode_id + 1}/{num_episodes}: "
                f"success={success}, steps={steps}, "
                f"task success rate={running_rate:.1%}"
            )
    finally:
        if video_writer is not None:
            video_writer.release()
        env.close()

    return {
        "task_id": task_id,
        "task_name": task.name,
        "instruction": instruction,
        "episodes": num_episodes,
        "successes": successes,
        "success_rate": successes / num_episodes,
        "episode_steps": episode_steps,
        "video_paths": video_paths,
    }


def evaluate_libero_goal(
    *,
    args: argparse.Namespace,
    model: ActionDiT,
    diffusion,
    normalizer: ActionNormalizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict:
    task_suite = create_libero_goal_suite()
    task_ids = range(task_suite.n_tasks) if args.task_id == -1 else [args.task_id]
    task_results = []

    for task_id in task_ids:
        task_results.append(
            evaluate_libero_task(
                task_suite=task_suite,
                task_id=task_id,
                model=model,
                diffusion=diffusion,
                normalizer=normalizer,
                device=device,
                amp_dtype=amp_dtype,
                eta=args.eta,
                num_episodes=args.num_episodes,
                max_steps=args.max_steps,
                wait_steps=args.wait_steps,
                execute_horizon=args.execute_horizon,
                seed=args.seed,
                render_gpu_device_id=args.render_gpu_device_id,
                video_dir=args.video_dir,
            )
        )

    total_episodes = sum(item["episodes"] for item in task_results)
    total_successes = sum(item["successes"] for item in task_results)
    total_success_rate = total_successes / total_episodes
    return {
        "suite": "libero_goal",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "ddim_steps": args.ddim_steps,
        "eta": args.eta,
        "execute_horizon": args.execute_horizon,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": total_success_rate,
        "tasks": task_results,
    }


def print_evaluation_summary(results: dict) -> None:
    print("\nLIBERO Goal evaluation summary")
    for task_result in results["tasks"]:
        print(
            f"  task {task_result['task_id']}: "
            f"{task_result['successes']}/{task_result['episodes']} "
            f"({task_result['success_rate']:.1%})"
        )
    print(
        f"overall: {results['total_successes']}/"
        f"{results['total_episodes']} "
        f"({results['total_success_rate']:.1%})"
    )


def save_results(results: dict, path: Path) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved results: {resolved}")


def make_dummy_inputs(
    device: torch.device,
    language: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Shape-only smoke-test data; these inputs have no task meaning."""
    states = torch.zeros(1, STATE_DIM, dtype=torch.float32, device=device)
    observations = torch.zeros(
        1, 3, 224, 224, dtype=torch.float32, device=device
    )
    return states, observations, [language]


def check_outputs(
    normalized_actions: torch.Tensor,
    raw_actions: torch.Tensor,
) -> None:
    expected_shape = (1, ACTION_CHUNK, ACTION_DIM)
    if tuple(normalized_actions.shape) != expected_shape:
        raise ValueError(
            f"Expected normalized actions {expected_shape}, "
            f"got {tuple(normalized_actions.shape)}"
        )
    if tuple(raw_actions.shape) != expected_shape:
        raise ValueError(
            f"Expected raw actions {expected_shape}, got {tuple(raw_actions.shape)}"
        )
    if not torch.isfinite(normalized_actions).all():
        raise ValueError("Normalized actions contain NaN or Inf")
    if not torch.isfinite(raw_actions).all():
        raise ValueError("Raw actions contain NaN or Inf")
    if not ((normalized_actions >= -1) & (normalized_actions <= 1)).all():
        raise ValueError("Normalized actions are outside [-1, 1]")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    checkpoint_path = require_file(args.checkpoint, "checkpoint")
    stats_path = require_file(args.stats_path, "action statistics")
    dino_repo = require_directory(args.dino_repo, "DINOv2 repository")
    dino_weights = require_file(args.dino_weights, "DINOv2 weights")
    qwen_model_path = require_directory(args.qwen_model_path, "Qwen model")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This model is intended to be tested on the CUDA training server."
        )
    device = torch.device("cuda")
    amp_dtype = precision_dtype(args.precision)

    print(f"device: {device}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"DDIM steps: {args.ddim_steps}, eta: {args.eta}")

    model = build_model(
        device=device,
        dino_repo=dino_repo,
        dino_weights=dino_weights,
        qwen_model_path=qwen_model_path,
        amp_dtype=amp_dtype,
    )
    load_model_weights(model, checkpoint_path)

    model.eval()

    # Keep the original 1000-step linear schedule used for training, but use
    # only a respaced subset during inference.
    diffusion = create_diffusion(
        timestep_respacing=f"ddim{args.ddim_steps}",
        noise_schedule="linear",
        diffusion_steps=1000,
        learn_sigma=LEARN_SIGMA,
    )
    normalizer = ActionNormalizer.from_stats_file(stats_path).to(device)

    if args.smoke_test:
        states, observations, language = make_dummy_inputs(device, args.language)
        normalized_actions, raw_actions = infer_action_chunk(
            model=model,
            diffusion=diffusion,
            normalizer=normalizer,
            states=states,
            observations=observations,
            language=language,
            eta=args.eta,
            amp_dtype=amp_dtype,
        )
        check_outputs(normalized_actions, raw_actions)

        print(f"normalized action shape: {tuple(normalized_actions.shape)}")
        print(
            "normalized action range: "
            f"[{normalized_actions.min().item():.6f}, "
            f"{normalized_actions.max().item():.6f}]"
        )
        print(f"raw action shape: {tuple(raw_actions.shape)}")
        print(f"first raw action: {raw_actions[0, 0].cpu().numpy()}")
        print("smoke test passed")
        return

    results = evaluate_libero_goal(
        args=args,
        model=model,
        diffusion=diffusion,
        normalizer=normalizer,
        device=device,
        amp_dtype=amp_dtype,
    )
    print_evaluation_summary(results)
    if args.results_path is not None:
        save_results(results, args.results_path)


if __name__ == "__main__":
    main()
