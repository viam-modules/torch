import torch.nn as nn
import torch
from inspector import Inspector
from torch.nn import functional as F
import unittest


class NotSequential1(nn.Module):
    def __init__(self):
        super(NotSequential1, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 1, 5)
        self.fully_connected = nn.Linear(70, 30)

        return super().__init_subclass__()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return F.softmax(self.fully_connected(x))


class NotSequential2(nn.Module):
    def __init__(self) -> None:
        super(NotSequential2, self).__init__()
        self.conv2 = nn.Conv2d(20, 1, 5)
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.fully_connected = nn.Linear(70, 30)

        return super().__init_subclass__()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return F.softmax(self.fully_connected(x))


class TestInputs(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.sequential_model = nn.Sequential(
            nn.Conv2d(1, 20, 5),
            nn.ReLU(),
            nn.Conv2d(20, 1, 5),
            nn.ReLU(),
            nn.Linear(70, 30),
        )

        self.not_sequential_model_1 = NotSequential1()
        self.not_sequential_model_2 = NotSequential2()
        self.inspector_seq = Inspector(self.not_sequential_model_1)
        self.inspector_not_seq_1 = Inspector(self.not_sequential_model_1)
        self.inspector_not_seq_2 = Inspector(self.not_sequential_model_2)

        self.valid_input_size = torch.Size([1, 9, 78])
        self.not_valid_input_size = torch.Size([1, 9, 9])

    def test_inference_seq(self):
        input_tensor = torch.ones(self.valid_input_size)
        _ = self.sequential_model(input_tensor)
        with self.assertRaises(RuntimeError):
            input_tensor = torch.ones(self.not_valid_input_size)
            _ = self.not_sequential_model_1(input_tensor)

    def test_inference_not_seq_1(self):
        input_tensor = torch.ones(self.valid_input_size)
        _ = self.not_sequential_model_1(input_tensor)
        with self.assertRaises(RuntimeError):
            input_tensor = torch.ones(self.not_valid_input_size)
            _ = self.not_sequential_model_1(input_tensor)

    def test_inference_not_seq_2(self):
        input_tensor = torch.ones(self.valid_input_size)
        _ = self.not_sequential_model_2(input_tensor)
        with self.assertRaises(RuntimeError):
            input_tensor = torch.ones(self.not_valid_input_size)
            _ = self.not_sequential_model_1(input_tensor)

    def test_found_input_shape_seq(self):
        found_input_shape = torch.Size(self.inspector_seq.reverse_module())
        self.assertEqual(found_input_shape, self.valid_input_size)
        self.assertNotEqual(found_input_shape, self.not_valid_input_size)

    def test_found_input_shape_not_seq_1(self):
        found_input_shape = torch.Size(self.inspector_not_seq_1.reverse_module())
        self.assertEqual(found_input_shape, self.valid_input_size)
        self.assertNotEqual(found_input_shape, self.not_valid_input_size)

    def test_found_input_shape_not_seq_2(self):
        found_input_shape = torch.Size(self.inspector_not_seq_2.reverse_module())
        self.assertNotEqual(found_input_shape, self.valid_input_size)
        self.assertNotEqual(found_input_shape, self.not_valid_input_size)


# Running the tests if this script is executed directly
if __name__ == "__main__":
    unittest.main()
