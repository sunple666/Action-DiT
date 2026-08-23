import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from data.libero_index import (
    EpisodeRef,
    LiberoSampleIndex,
    SplitManifest,
    build_split_manifest,
)


def create_demo_file(path: Path, lengths: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        data.attrs["problem_info"] = json.dumps(
            {"language_instruction": "test instruction"}
        )
        for demo_index, length in enumerate(lengths):
            demo = data.create_group(f"demo_{demo_index}")
            demo.create_dataset("actions", data=np.zeros((length, 7)))
            obs = demo.create_group("obs")
            obs.create_dataset(
                "agentview_rgb", data=np.zeros((length, 128, 128, 3), dtype=np.uint8)
            )
            obs.create_dataset("ee_states", data=np.zeros((length, 6)))
            obs.create_dataset("gripper_states", data=np.zeros((length, 2)))


class LiberoIndexTest(unittest.TestCase):
    def test_global_index_crosses_episode_boundary(self) -> None:
        episodes = (
            EpisodeRef("task.hdf5", "demo_0", 3, "task"),
            EpisodeRef("task.hdf5", "demo_1", 5, "task"),
        )
        index = LiberoSampleIndex(episodes)

        self.assertEqual(len(index), 8)
        self.assertEqual(index.locate(2).episode.demo, "demo_0")
        self.assertEqual(index.locate(2).timestep, 2)
        self.assertEqual(index.locate(3).episode.demo, "demo_1")
        self.assertEqual(index.locate(3).timestep, 0)
        self.assertEqual(index.locate(-1).timestep, 4)

    def test_split_is_disjoint_reproducible_and_portable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            create_demo_file(root / "libero_goal" / "task.hdf5", [2] * 10)

            first = build_split_manifest(root, val_ratio=0.2, seed=42)
            second = build_split_manifest(root, val_ratio=0.2, seed=42)

            self.assertEqual(first, second)
            self.assertEqual(len(first.train), 8)
            self.assertEqual(len(first.val), 2)
            train_keys = {(item.file, item.demo) for item in first.train}
            val_keys = {(item.file, item.demo) for item in first.val}
            self.assertTrue(train_keys.isdisjoint(val_keys))
            self.assertTrue(
                all(item.file == "libero_goal/task.hdf5" for item in first.train)
            )

            manifest_path = root / "split.json"
            first.save(manifest_path)
            self.assertEqual(SplitManifest.load(manifest_path), first)


if __name__ == "__main__":
    unittest.main()
