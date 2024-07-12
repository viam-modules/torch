"Module for unit testing functionalities related to TorchModel and TorchMLModelModule."

from google.protobuf.struct_pb2 import Struct #pylint: disable=(no-name-in-module)

import unittest
from torchvision.models.detection.rpn import AnchorGenerator

from torchvision.models.detection import FasterRCNN

from torchvision.models import MobileNet_V2_Weights
import torchvision
from model.model import TorchModel
from model_inspector.inspector import Inspector
from torch_mlmodel_module import TorchMLModelModule
from viam.services.mlmodel import Metadata
from viam.proto.app.robot import ComponentConfig

from typing import Any, Mapping
import numpy as np
import os



import torch



def make_component_config(dictionary: Mapping[str, Any]) -> ComponentConfig:
    " makes a mock config"
    struct = Struct()
    struct.update(dictionary)
    return ComponentConfig(attributes=struct)


config = (
    make_component_config({"model_path": "model path"}),
    "received only one dimension attribute",
)


class TestInputs(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for validating TorchModel and TorchMLModelModule functionalities.
    """
    @staticmethod
    def load_resnet_weights():
        """
        Load ResNet weights from a serialized file.
        """
        return TorchModel(
            path_to_serialized_file=os.path.join(
                "examples", "resnet_18", "resnet18-f37072fd.pth"
            )
        )

    @staticmethod
    def load_standalone_resnet():
        """
        Load a standalone ResNet model.
        """
        return TorchModel(
            path_to_serialized_file=os.path.join(
                "examples", "resnet_18_scripted", "resnet-18.pt"
            )
        )

    @staticmethod
    def load_detector_from_torchvision():
        """
        Load a detector model using torchvision.
        """
        backbone = torchvision.models.mobilenet_v2(
            weights=MobileNet_V2_Weights.DEFAULT
        ).features
        backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0"], output_size=7, sampling_ratio=2
        )
        model = FasterRCNN(
            backbone,
            num_classes=2,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )

        model.eval()
        return TorchModel(path_to_serialized_file=None, model=model)

    def __init__(self, methodName: str = "runTest") -> None: #pylint: disable=(useless-parent-delegation)
        super().__init__(methodName)

    async def test_validate(self):
        """
        Test validation of configuration using TorchMLModelModule.
        """
        response = TorchMLModelModule.validate_config(config=config[0])
        self.assertEqual(response, [])

    async def test_validate_empty_config(self):
        """
        Test validation with an empty configuration.
        """
        empty_config = make_component_config({})
        with self.assertRaises(Exception) as excinfo:
            await TorchMLModelModule.validate_config(config=empty_config)

        self.assertIn(
            "model_path can't be empty. model is required for torch mlmoded service module.",
            str(excinfo.exception),
        )

    def test_error_loading_weights(self):
        """
        Test error handling when loading ResNet weights.
        """
        with self.assertRaises(TypeError):
            _ = self.load_resnet_weights()

    def test_resnet_metadata(self):
        """
        Test metadata retrieval for ResNet model.
        """
        model: TorchModel = self.load_standalone_resnet()
        x = torch.ones(3, 300, 400).unsqueeze(0)
        output = model.infer({"any_input_name_you_want": x.numpy()})
        self.assertIsInstance(output, dict)
        inspector = Inspector(model)
        metadata: Metadata = inspector.find_metadata(label_path="fake_path")

        for output_name, output in output.items():
            output_checked = False
            for output_info in metadata.output_info:
                if output_info.name == output_name:
                    self.assertEqual(
                        output.shape[0], output_info.shape[0]
                    )  # check at index 0 because one is (1000,) and  the other is [1000]
                    output_checked = True
                    print(f"Checked {output_name} ")
            self.assertTrue(output_checked)

    def test_detector_metadata(self):
        """
        Test metadata retrieval for detector model.
        """
        model: TorchModel = self.load_detector_from_torchvision()
        x = torch.ones(3, 300, 400).unsqueeze(0)
        output = model.infer({"any_input_name_you_want": x.numpy()})
        self.assertIsInstance(output, dict)
        inspector = Inspector(model)
        metadata: Metadata = inspector.find_metadata(label_path="fake_path")

        for output_name, output in output.items():
            output_checked = False
            for output_info in metadata.output_info:
                if output_info.name == output_name:
                    self.assertEqual(
                        output.shape[0], output_info.shape[0]
                    )  # check at index 0 because one is (1000,) and  the other is [1000]
                    output_checked = True
                    print(f"Checked {output_name} ")
            self.assertTrue(output_checked)

    def test_infer_method(self):
        """
        Test inference method of the detector model.
        """
        model: TorchModel = self.load_detector_from_torchvision()
        x = torch.ones(3, 300, 400).unsqueeze(0)
        output = model.infer({"input_name": x.numpy()})
        self.assertIsInstance(output, dict)

        # Assert the structure of the output based on wrap_output function
        for key, value in output.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, np.ndarray)


if __name__ == "__main__":
    unittest.main()
    