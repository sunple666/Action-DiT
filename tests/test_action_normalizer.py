import tempfile
import unittest
from pathlib import Path

import torch

from data.action_normalizer import ActionNormalizer
from data.action_stats import ActionStats, STATS_VERSION


class ActionNormalizerTest(unittest.TestCase):
    def test_endpoints_and_round_trip(self) -> None:
        normalizer = ActionNormalizer(
            action_min=[-2.0, 0.0],
            action_max=[2.0, 10.0],
        )
        raw = torch.tensor([[-2.0, 0.0], [0.0, 5.0], [2.0, 10.0]])
        normalized = normalizer.normalize(raw)

        torch.testing.assert_close(
            normalized,
            torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]),
        )
        torch.testing.assert_close(normalizer.denormalize(normalized), raw)

    def test_load_from_stats_file(self) -> None:
        stats = ActionStats(
            version=STATS_VERSION,
            split="train",
            count=3,
            action_dim=2,
            action_min=(-1.0, -2.0),
            action_max=(1.0, 2.0),
            action_mean=(0.0, 0.0),
            action_std=(1.0, 1.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            stats.save(path)
            normalizer = ActionNormalizer.from_stats_file(path)
            torch.testing.assert_close(
                normalizer.action_min, torch.tensor([-1.0, -2.0])
            )
            torch.testing.assert_close(
                normalizer.action_max, torch.tensor([1.0, 2.0])
            )


if __name__ == "__main__":
    unittest.main()
