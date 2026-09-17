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
                wrist_images = np.zeros_like(images)
                wrist_images[..., 1] = np.arange(
                    length, dtype=np.uint8
                )[:, None, None]
                obs.create_dataset("eye_in_hand_rgb", data=wrist_images)
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

        self.assertEqual(len(dataset), 19)
        self.assertEqual(first["action"].shape, (16, 7))
        self.assertTrue(first["action_mask"].all())
        self.assertEqual(first["action_transition_mask"].shape, (16,))
        self.assertEqual(first["action_post_transition_mask"].shape, (16,))
        self.assertEqual(first["state"].shape, (8,))
        self.assertEqual(first["agentview_observation"].shape, (3, 224, 224))
        self.assertEqual(first["wrist_observation"].shape, (3, 224, 224))
        self.assertTrue(torch.isfinite(first["agentview_observation"]).all())
        self.assertTrue(torch.isfinite(first["wrist_observation"]).all())
        self.assertEqual(first["text"], "test task")

        self.assertEqual(int(tail["action_mask"].sum()), 1)
        self.assertTrue(torch.equal(tail["action"][1:], torch.zeros(15, 7)))
        torch.testing.assert_close(
        first["action"][0],
        torch.arange(7, 14, dtype=torch.float32),
        )
        torch.testing.assert_close(
        tail["action"][0],
        torch.arange(133, 140, dtype=torch.float32),
        )
        dataset.close()

    def test_transition_masks_and_sampling_weights(self) -> None:
        file_path = self.root / self.relative_file
        with h5py.File(file_path, "r+") as file:
            actions = file["data/demo_0/actions"]
            actions[:, -1] = 1.0
            actions[0:2, -1] = -1.0

        dataset = ActionDataset(
            self.root,
            self.manifest_path,
            "train",
            post_transition_steps=2,
        )
        sample = dataset[0]
        expected_transition = torch.zeros(16, dtype=torch.bool)
        expected_transition[1] = True
        expected_post = torch.zeros(16, dtype=torch.bool)
        expected_post[2:4] = True
        self.assertTrue(
            torch.equal(sample["action_transition_mask"], expected_transition)
        )
        self.assertTrue(
            torch.equal(sample["action_post_transition_mask"], expected_post)
        )

        post_sample = dataset[2]
        expected_post_at_chunk_start = torch.zeros(16, dtype=torch.bool)
        expected_post_at_chunk_start[:2] = True
        self.assertFalse(post_sample["action_transition_mask"].any())
        self.assertTrue(
            torch.equal(
                post_sample["action_post_transition_mask"],
                expected_post_at_chunk_start,
            )
        )

        weights, near_count = dataset.transition_sampling_weights(
            transition_window=1,
            oversample_factor=4.0,
        )
        self.assertEqual(near_count, 3)
        torch.testing.assert_close(
            weights[:3], torch.full((3,), 4.0, dtype=torch.double)
        )
        torch.testing.assert_close(
            weights[3:], torch.ones(16, dtype=torch.double)
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
