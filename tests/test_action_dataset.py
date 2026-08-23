import json
import pickle
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from data.action_dataset import ActionDataset
from data.libero_index import EpisodeRef, SplitManifest


class ActionDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.relative_file = "libero_goal/test_demo.hdf5"
        file_path = self.root / self.relative_file
        file_path.parent.mkdir(parents=True)

        with h5py.File(file_path, "w") as file:
            data = file.create_group("data")
            data.attrs["problem_info"] = json.dumps(
                {"language_instruction": "test task"}
            )
            for demo_index, length in enumerate((20, 5)):
                demo = data.create_group(f"demo_{demo_index}")
                actions = np.arange(length * 7, dtype=np.float32).reshape(length, 7)
                demo.create_dataset("actions", data=actions)
                obs = demo.create_group("obs")
                images = np.zeros((length, 128, 128, 3), dtype=np.uint8)
                images[..., 0] = np.arange(length, dtype=np.uint8)[:, None, None]
                obs.create_dataset("agentview_rgb", data=images)
                ee_states = np.arange(length * 6, dtype=np.float32).reshape(length, 6)
                obs.create_dataset("ee_states", data=ee_states)
                obs.create_dataset(
                    "gripper_states", data=np.ones((length, 2), dtype=np.float32)
                )

        train_episode = EpisodeRef(self.relative_file, "demo_0", 20, "test task")
        val_episode = EpisodeRef(self.relative_file, "demo_1", 5, "test task")
        self.manifest_path = self.root / "manifest.json"
        SplitManifest(
            seed=42,
            val_ratio=0.5,
            train=(train_episode,),
            val=(val_episode,),
        ).save(self.manifest_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_full_and_tail_action_chunks(self) -> None:
        dataset = ActionDataset(self.root, self.manifest_path, "train")
        first = dataset[0]
        tail = dataset[-1]

        self.assertEqual(len(dataset), 20)
        self.assertEqual(first["action"].shape, (16, 7))
        self.assertTrue(first["action_mask"].all())
        self.assertEqual(first["state"].shape, (8,))
        self.assertEqual(first["observation"].shape, (3, 224, 224))
        self.assertTrue(torch.isfinite(first["observation"]).all())
        self.assertEqual(first["text"], "test task")

        self.assertEqual(int(tail["action_mask"].sum()), 1)
        self.assertTrue(torch.equal(tail["action"][1:], torch.zeros(15, 7)))
        torch.testing.assert_close(
            tail["action"][0], torch.arange(133, 140, dtype=torch.float32)
        )
        dataset.close()

    def test_open_handles_are_not_pickled(self) -> None:
        dataset = ActionDataset(self.root, self.manifest_path, "train")
        dataset[0]
        self.assertEqual(len(dataset._open_files), 1)

        restored = pickle.loads(pickle.dumps(dataset))
        self.assertEqual(len(dataset._open_files), 1)
        self.assertEqual(len(restored._open_files), 0)
        dataset.close()
        restored.close()


if __name__ == "__main__":
    unittest.main()
