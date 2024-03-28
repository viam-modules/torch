import torch
import unittest
from src.model.model import TorchModel
from src.model_inspector.inspector import Inspector
from viam.services.mlmodel import Metadata
from torchvision.models.detection import FasterRCNN

from torchvision.models import MobileNet_V2_Weights
import torchvision
import os
from torchvision.models.detection.rpn import AnchorGenerator


class TestInputs(unittest.TestCase):
    @staticmethod
    def load_resnet_weights():
        return TorchModel(
            path_to_serialized_file=os.path.join(
                "examples", "resnet_18", "resnet18-f37072fd.pth"
            )
        )

    @staticmethod
    def load_standalone_resnet():
        return TorchModel(
            path_to_serialized_file=os.path.join(
                "examples", "resnet_18_scripted", "resnet-18.pt"
            )
        )

    @staticmethod
    def load_detector_from_torchvision():
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

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_error_loading_weights(self):
        with self.assertRaises(TypeError):
            _ = self.load_resnet_weights()

    def test_resnet_metadata(self):
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


if __name__ == "__main__":
    unittest.main()
