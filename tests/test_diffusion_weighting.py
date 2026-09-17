import unittest

import torch

from diffusion.gaussian_diffusion import mean_flat, weighted_mean_flat


class DiffusionWeightingTest(unittest.TestCase):
    def test_uniform_valid_weights_match_original_masked_mean(self) -> None:
        values = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
        mask = torch.tensor([[True, False], [True, True]])
        weights = mask.unsqueeze(-1).expand_as(values).float()
        torch.testing.assert_close(
            weighted_mean_flat(values, weights),
            mean_flat(values, mask=mask),
        )

    def test_weighted_mean_is_normalized_per_sample(self) -> None:
        values = torch.tensor(
            [
                [[1.0, 3.0], [5.0, 7.0]],
                [[2.0, 4.0], [6.0, 8.0]],
            ]
        )
        weights = torch.tensor(
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 3.0]],
            ]
        )
        actual = weighted_mean_flat(values, weights)
        expected = torch.tensor([2.0, 7.5])
        torch.testing.assert_close(actual, expected)

    def test_rejects_zero_weight_sample(self) -> None:
        values = torch.ones(2, 2, 2)
        weights = torch.ones_like(values)
        weights[1] = 0.0
        with self.assertRaisesRegex(ValueError, "positive total weight"):
            weighted_mean_flat(values, weights)


if __name__ == "__main__":
    unittest.main()
