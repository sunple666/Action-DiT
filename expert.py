"""Replay recorded LIBERO expert demonstrations without loading ActionDiT.

This diagnostic separates policy failures from dataset / environment failures.
For each selected HDF5 demonstration it:

1. restores the demonstration MuJoCo XML and ``states[0]``;
2. executes the recorded actions in their original order;
3. checks task success;
4. compares the replayed simulator state after ``actions[t]`` with
   ``states[t + 1]``; and
5. compares the observation returned after ``actions[t]`` with the saved
   ``obs[t]``.

The last comparison is especially important for ActionDiT: if saved ``obs[t]``
matches the post-action observation, then pairing it with ``actions[t]`` for
causal policy training leaks the result of the action into the condition.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Headless-server default. An explicit user setting still wins.
os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import numpy as np

from data.libero_index import EpisodeRef, SplitManifest


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_STORAGE_ROOT = Path("/root/autodl-tmp/actiondit_storage")
RUNTIME_STORAGE_ROOT = Path(
    os.environ.get("STORE", str(DEFAULT_STORAGE_ROOT))
).expanduser()
ACTION_DIM = 7
VIDEO_RESOLUTION = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay expert actions from LIBERO HDF5 demonstrations and check "
            "success, state transitions, and observation/action alignment."
        )
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=RUNTIME_STORAGE_ROOT / "datasets" / "libero",
    )
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=PROJECT_ROOT / "configs" / "libero_goal_split.json",
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--task_id",
        type=int,
        default=0,
        help="LIBERO Goal task id in [0, 9], or -1 for all ten tasks.",
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        default=3,
        help="Number of demonstrations replayed per selected task.",
    )
    parser.add_argument(
        "--demo",
        type=str,
        default=None,
        help="Optional exact demo name such as demo_0; requires one task.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Maximum recorded actions per demo; 0 executes the full demo.",
    )
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument(
        "--render_gpu_device_id",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--state_tolerance",
        type=float,
        default=0.01,
        help="L2 threshold used to count divergent replayed simulator states.",
    )
    parser.add_argument(
        "--no_model_xml",
        action="store_true",
        help=(
            "Use only the benchmark BDDL environment instead of restoring the "
            "XML stored with each demonstration. Intended only as a fallback."
        ),
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=None,
        help="Optional directory for one MP4 per replayed demonstration.",
    )
    parser.add_argument(
        "--results_path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "expert_replay.json",
    )
    args = parser.parse_args()

    if not -1 <= args.task_id <= 9:
        parser.error("--task_id must be in [0, 9], or -1 for all tasks")
    if args.num_demos <= 0:
        parser.error("--num_demos must be positive")
    if args.max_steps < 0:
        parser.error("--max_steps cannot be negative")
    if args.render_gpu_device_id < 0:
        parser.error("--render_gpu_device_id cannot be negative")
    if args.state_tolerance <= 0:
        parser.error("--state_tolerance must be positive")
    if args.demo is not None and args.task_id == -1:
        parser.error("--demo requires a concrete --task_id")
    return args


def require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def require_directory(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"{description} does not exist: {resolved}")
    return resolved


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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


def decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def get_core_env(env):
    """Return the robosuite environment wrapped by OffScreenRenderEnv."""
    return getattr(env, "env", env)


def restore_demo_state(
    env,
    *,
    initial_state: np.ndarray,
    model_xml: str | None,
) -> dict:
    """Restore the exact demonstration model and its first recorded state."""
    env.reset()
    core_env = get_core_env(env)

    if model_xml is not None:
        try:
            from libero.libero.utils.utils import postprocess_model_xml
        except ImportError as exc:
            raise ImportError("LIBERO XML post-processing is unavailable") from exc

        processed_xml = postprocess_model_xml(model_xml, {})
        core_env.reset_from_xml_string(processed_xml)
        core_env.sim.reset()

    # LIBERO's wrapper refreshes observables after setting the flattened state.
    obs = env.set_init_state(initial_state)
    return obs


def current_flattened_state(env) -> np.ndarray:
    state = get_core_env(env).sim.get_state().flatten()
    return np.asarray(state, dtype=np.float64)


def render_frame(env) -> np.ndarray:
    frame = get_core_env(env).sim.render(
        width=VIDEO_RESOLUTION,
        height=VIDEO_RESOLUTION,
        camera_name="agentview",
    )
    return np.flipud(np.asarray(frame))


def saved_observation_errors(
    obs: dict,
    saved_obs,
    index: int,
) -> dict[str, float]:
    """Compare a post-step environment observation to saved processed data."""
    from robosuite.utils import transform_utils

    errors: dict[str, float] = {}

    if "agentview_rgb" in saved_obs and "agentview_image" in obs:
        replayed = np.asarray(obs["agentview_image"], dtype=np.float32)
        saved = np.asarray(saved_obs["agentview_rgb"][index], dtype=np.float32)
        if replayed.shape == saved.shape:
            errors["image_mae_0_to_1"] = float(
                np.mean(np.abs(replayed - saved)) / 255.0
            )

    if "ee_states" in saved_obs:
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
        axis_angle = np.asarray(
            transform_utils.quat2axisangle(eef_quat.copy()),
            dtype=np.float32,
        )
        replayed_ee = np.concatenate((eef_pos, axis_angle))
        saved_ee = np.asarray(saved_obs["ee_states"][index], dtype=np.float32)
        errors["ee_l2"] = float(np.linalg.norm(replayed_ee - saved_ee))

    if "gripper_states" in saved_obs:
        replayed_gripper = np.asarray(
            obs["robot0_gripper_qpos"], dtype=np.float32
        )
        saved_gripper = np.asarray(
            saved_obs["gripper_states"][index], dtype=np.float32
        )
        errors["gripper_l2"] = float(
            np.linalg.norm(replayed_gripper - saved_gripper)
        )

    return errors


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def open_video_writer(video_path: Path):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Video recording requires OpenCV (cv2)") from exc

    video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        20.0,
        (VIDEO_RESOLUTION, VIDEO_RESOLUTION),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {video_path}")
    return cv2, writer


def replay_episode(
    *,
    env,
    dataset_root: Path,
    episode: EpisodeRef,
    task_id: int,
    max_steps: int,
    state_tolerance: float,
    use_model_xml: bool,
    video_dir: Path | None,
) -> dict[str, Any]:
    hdf5_path = require_file(dataset_root / episode.file, "demonstration file")
    with h5py.File(hdf5_path, "r") as file:
        demo = file["data"][episode.demo]
        if "states" not in demo or "actions" not in demo:
            raise KeyError(f"{hdf5_path}:{episode.demo} lacks states or actions")

        states = np.asarray(demo["states"], dtype=np.float64)
        actions = np.asarray(demo["actions"], dtype=np.float32)
        if states.ndim != 2 or states.shape[0] != actions.shape[0]:
            raise ValueError(
                f"Invalid states/actions shapes in {episode.demo}: "
                f"{states.shape} and {actions.shape}"
            )
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(f"Expected actions [T, 7], got {actions.shape}")

        model_xml = None
        if use_model_xml:
            if "model_file" not in demo.attrs:
                raise KeyError(f"{episode.demo} has no model_file attribute")
            model_xml = decode_text(demo.attrs["model_file"])

        obs_before_action = restore_demo_state(
            env,
            initial_state=states[0],
            model_xml=model_xml,
        )
        restored_state = current_flattened_state(env)
        if restored_state.shape != states[0].shape:
            raise ValueError(
                "Restored and saved initial states have different shapes: "
                f"{restored_state.shape} vs {states[0].shape}"
            )
        initial_state_error = float(np.linalg.norm(restored_state - states[0]))

        cv2 = None
        writer = None
        video_path = None
        if video_dir is not None:
            video_path = (
                video_dir
                / f"task_{task_id:02d}_{episode.demo}_expert.mp4"
            )
            cv2, writer = open_video_writer(video_path)
            writer.write(np.ascontiguousarray(render_frame(env)[..., ::-1]))

        limit = len(actions) if max_steps == 0 else min(max_steps, len(actions))
        state_l2_errors: list[float] = []
        state_max_abs_errors: list[float] = []
        pre_image_errors: list[float] = []
        pre_ee_errors: list[float] = []
        pre_gripper_errors: list[float] = []
        image_errors: list[float] = []
        ee_errors: list[float] = []
        gripper_errors: list[float] = []
        first_success_step: int | None = None
        final_reward = 0.0
        final_done = False

        try:
            for step in range(limit):
                pre_observation_errors = saved_observation_errors(
                    obs_before_action,
                    demo["obs"],
                    step,
                )
                if "image_mae_0_to_1" in pre_observation_errors:
                    pre_image_errors.append(
                        pre_observation_errors["image_mae_0_to_1"]
                    )
                if "ee_l2" in pre_observation_errors:
                    pre_ee_errors.append(pre_observation_errors["ee_l2"])
                if "gripper_l2" in pre_observation_errors:
                    pre_gripper_errors.append(
                        pre_observation_errors["gripper_l2"]
                    )

                obs, reward, done, _ = env.step(actions[step].tolist())
                final_reward = float(reward)
                final_done = bool(done)

                if writer is not None:
                    writer.write(
                        np.ascontiguousarray(render_frame(env)[..., ::-1])
                    )

                success_now = bool(done) or bool(reward > 0) or bool(
                    env.check_success()
                )
                if success_now and first_success_step is None:
                    first_success_step = step + 1

                # No state exists for the transition after the final action.
                if step + 1 < len(states):
                    replayed_state = current_flattened_state(env)
                    target_state = states[step + 1]
                    if replayed_state.shape != target_state.shape:
                        raise ValueError(
                            "Replayed and saved simulator states have different "
                            f"shapes: {replayed_state.shape} vs {target_state.shape}"
                        )
                    difference = replayed_state - target_state
                    state_l2_errors.append(float(np.linalg.norm(difference)))
                    state_max_abs_errors.append(float(np.max(np.abs(difference))))

                observation_errors = saved_observation_errors(
                    obs,
                    demo["obs"],
                    step,
                )
                if "image_mae_0_to_1" in observation_errors:
                    image_errors.append(observation_errors["image_mae_0_to_1"])
                if "ee_l2" in observation_errors:
                    ee_errors.append(observation_errors["ee_l2"])
                if "gripper_l2" in observation_errors:
                    gripper_errors.append(observation_errors["gripper_l2"])
                obs_before_action = obs
        finally:
            if writer is not None:
                writer.release()

    divergent_states = sum(
        error > state_tolerance for error in state_l2_errors
    )
    success = first_success_step is not None or bool(env.check_success())
    return {
        "episode": asdict(episode),
        "task_id": task_id,
        "actions_executed": limit,
        "success": success,
        "first_success_step": first_success_step,
        "final_reward": final_reward,
        "final_done": final_done,
        "initial_state_l2_error": initial_state_error,
        "state_transition_l2": summarize(state_l2_errors),
        "state_transition_max_abs": summarize(state_max_abs_errors),
        "state_tolerance": state_tolerance,
        "divergent_state_transitions": divergent_states,
        "pre_action_observation": {
            "image_mae_0_to_1": summarize(pre_image_errors),
            "ee_l2": summarize(pre_ee_errors),
            "gripper_l2": summarize(pre_gripper_errors),
        },
        "post_action_observation": {
            "image_mae_0_to_1": summarize(image_errors),
            "ee_l2": summarize(ee_errors),
            "gripper_l2": summarize(gripper_errors),
        },
        "video_path": str(video_path) if video_path is not None else None,
    }


def select_episodes(
    *,
    manifest: SplitManifest,
    split: str,
    suite,
    task_ids: list[int],
    num_demos: int,
    demo_name: str | None,
) -> dict[int, list[EpisodeRef]]:
    split_episodes = manifest.episodes(split)
    selected: dict[int, list[EpisodeRef]] = {}

    for task_id in task_ids:
        instruction = suite.get_task(task_id).language
        candidates = [
            episode
            for episode in split_episodes
            if episode.instruction == instruction
        ]
        if demo_name is not None:
            candidates = [
                episode for episode in candidates if episode.demo == demo_name
            ]
        if not candidates:
            suffix = f" and demo {demo_name}" if demo_name else ""
            raise ValueError(
                f"No {split} episode matches task {task_id}{suffix}: "
                f"{instruction}"
            )
        selected[task_id] = candidates[:num_demos]
    return selected


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    dataset_root = require_directory(args.dataset_root, "dataset root")
    manifest_path = require_file(args.manifest_path, "split manifest")
    manifest = SplitManifest.load(manifest_path)
    suite = create_libero_goal_suite()
    task_ids = list(range(10)) if args.task_id == -1 else [args.task_id]
    selected = select_episodes(
        manifest=manifest,
        split=args.split,
        suite=suite,
        task_ids=task_ids,
        num_demos=args.num_demos,
        demo_name=args.demo,
    )

    try:
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as exc:
        raise ImportError(
            "LIBERO environments are unavailable in this Python environment"
        ) from exc

    video_dir = None
    if args.video_dir is not None:
        video_dir = args.video_dir.expanduser().resolve()
        video_dir.mkdir(parents=True, exist_ok=True)

    episode_results: list[dict[str, Any]] = []
    for task_id in task_ids:
        task = suite.get_task(task_id)
        print(f"\ntask {task_id}: {task.language}")
        env = OffScreenRenderEnv(
            bddl_file_name=suite.get_task_bddl_file_path(task_id),
            camera_heights=128,
            camera_widths=128,
            render_gpu_device_id=args.render_gpu_device_id,
        )
        env.seed(args.seed)
        try:
            for episode in selected[task_id]:
                result = replay_episode(
                    env=env,
                    dataset_root=dataset_root,
                    episode=episode,
                    task_id=task_id,
                    max_steps=args.max_steps,
                    state_tolerance=args.state_tolerance,
                    use_model_xml=not args.no_model_xml,
                    video_dir=video_dir,
                )
                episode_results.append(result)
                state_mean = result["state_transition_l2"]["mean"]
                pre_image_mean = result["pre_action_observation"][
                    "image_mae_0_to_1"
                ]["mean"]
                image_mean = result["post_action_observation"][
                    "image_mae_0_to_1"
                ]["mean"]
                print(
                    f"  {episode.demo}: success={result['success']} "
                    f"steps={result['actions_executed']} "
                    f"state_l2_mean={state_mean!r} "
                    f"image_mae_pre={pre_image_mean!r} "
                    f"image_mae_post={image_mean!r} "
                    f"diverged={result['divergent_state_transitions']}"
                )
        finally:
            env.close()

    successes = sum(int(result["success"]) for result in episode_results)
    total = len(episode_results)
    total_state_comparisons = sum(
        result["state_transition_l2"]["count"] for result in episode_results
    )
    total_divergences = sum(
        result["divergent_state_transitions"] for result in episode_results
    )
    output = {
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "split": args.split,
        "task_ids": task_ids,
        "num_episodes": total,
        "successes": successes,
        "success_rate": successes / total,
        "state_tolerance": args.state_tolerance,
        "state_transition_comparisons": total_state_comparisons,
        "divergent_state_transitions": total_divergences,
        "state_divergence_rate": (
            total_divergences / total_state_comparisons
            if total_state_comparisons
            else None
        ),
        "used_demonstration_model_xml": not args.no_model_xml,
        "episodes": episode_results,
        "interpretation": {
            "expert_success": (
                "If expert replay succeeds, dataset actions and the LIBERO "
                "controller are compatible. If it fails with large state "
                "transition errors, diagnose environment/XML/controller setup."
            ),
            "observation_alignment": (
                "Compare pre_action_observation with post_action_observation. "
                "Whichever has much smaller errors identifies when saved obs[t] "
                "was collected. If post-action is smaller, pairing obs[t] with "
                "action[t] leaks the result of that action into policy input."
            ),
        },
    }

    results_path = args.results_path.expanduser().resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"\nexpert replay: {successes}/{total} "
        f"({successes / total:.1%})"
    )
    print(
        f"state divergence: {total_divergences}/{total_state_comparisons} "
        f"at tolerance {args.state_tolerance}"
    )
    print(f"saved results: {results_path}")


if __name__ == "__main__":
    main()
