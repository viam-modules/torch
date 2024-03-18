import torch.nn as nn
import torch
from inspector import Inspector
from torch.nn import functional as F
import unittest
from input_tester import InputTester


class TestInputs(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.image_model = nn.Sequential(
            nn.Conv2d(1, 20, 5),
            nn.ReLU(),
            nn.Conv2d(20, 1, 5),
            nn.ReLU(),
        )

        self.nc = InputTester(self.image_model)

    def test_grey_image_model(self):
        grey_image_model = nn.Sequential(
            nn.Conv2d(1, 20, 5),
            nn.ReLU(),
            nn.Conv2d(20, 1, 5),
            nn.ReLU(),
        )
        input_tester = InputTester(grey_image_model)
        input_shape = input_tester.get_shapes()
        self.assertEqual(input_shape, [1, -1, -1])

    def test_rgb_image_model(self):
        grey_image_model = nn.Sequential(
            nn.Conv2d(3, 20, 5),
            nn.ReLU(),
            nn.Conv2d(20, 1, 5),
            nn.ReLU(),
        )
        input_tester = InputTester(grey_image_model)
        input_shape = input_tester.get_shapes()
        self.assertEqual(input_shape, [3, -1, -1])

    def test_mono_audio_model(self):
        grey_image_model = nn.Sequential(
            nn.Conv1d(1, 3, 5),
            nn.ReLU(),
        )
        input_tester = InputTester(grey_image_model)
        input_shape = input_tester.get_shapes()
        self.assertEqual(input_shape, [1, -1])

    def test_stereo_audio_model(self):
        grey_image_model = nn.Sequential(
            nn.Conv1d(2, 3, 3),
            nn.ReLU(),
        )
        input_tester = InputTester(grey_image_model)
        input_shape = input_tester.get_shapes()
        self.assertEqual(input_shape, [2, -1])


if __name__ == "__main__":
    unittest.main()
