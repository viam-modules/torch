"""
Module for inspecting and gathering metadata from a PyTorch model.
This module provides functionality to inspect a PyTorch model,
reverse its layers for input shape calculation, validate input shapes,
and retrieve metadata such as input and output tensor shapes.
"""
from viam.services.mlmodel import Metadata, TensorInfo
from viam.logging import getLogger
from viam.utils import dict_to_struct

from model_inspector.input_size_calculator import InputSizeCalculator
from model_inspector.input_tester import InputTester

from torch import nn
import torch


LOGGER = getLogger(__name__)


class Inspector:
    "Inspector class for analyzing and gathering metadata from a PyTorch model."

    def __init__(self, model: nn.Module) -> None:
        # self.summary: ModelStatistics = summary(module, input_size=[1,3, 640,480])
        self.model = model
        self.input_size_calculator = InputSizeCalculator()

        self.dimensionality = None
        input_shape_candidate = self.reverse_module()
        self.input_tester = InputTester(self.model, input_shape_candidate)

    def find_metadata(self, label_path: str):
        """
        Gather metadata including input and output tensor information.

        Args:
            label_path (str): Path to the label file, if available.

        Returns:
            Metadata: Metadata object containing model information.
        """
        input_info, output_info = [], []
        input_shapes, output_shapes = self.input_tester.get_shapes()
        for input_tensor_name, shape in input_shapes.items():
            input_info.append(TensorInfo(name=input_tensor_name, shape=shape))

        for output_tensor_name, shape in output_shapes.items():
            output_info.append(
                TensorInfo(
                    name=output_tensor_name,
                    shape=shape,
                    extra=dict_to_struct({"label": label_path})
                    if (label_path is not None)
                    else None,
                )
            )

        return Metadata(
            name="torch-cpu", input_info=input_info, output_info=output_info
        )

    def reverse_module(self):
        """
        Reverse a nn.Module.
        If the self.model is of type torch.nn.Sequential, self.model.children()
        returns its layers in reverse orders.
        Else, it will return layers in the order they were instantiated,
        which might be different from the order of execution. (see test_input_size_calculator.py)

        Returns:
            List[int]: input shape candidate.
        """
        modules = list(self.model.model.children())
        modules.reverse()  # last layer comes first so need to reverse it

        # if self.len_module_labels is not None:
        #     output_shape = (1, self.len_module_labels)
        # else:
        #     output_shape = None

        ##Reverse model
        input_shape = None
        for module in modules:
            if module is not None:
                output_shape = self.input_size_calculator.get_input_size(
                    module, input_shape
                )
                LOGGER.info(
                    "For module %s, the output shape is %s", module, output_shape
                )
            else:
                continue  # sometimes some children are None

        return input_shape

    def is_valid_input_shape(self, input_shape):
        """
        Validate if a given input shape is valid for the model.

        Args:
            input_shape: Shape of the input tensor to validate.

        Returns:
            torch.Size or None: Size of the output tensor if input shape is valid, or None if not.
        """
        input_tensor = torch.ones(input_shape)
        try:
            output = self.model(input_tensor)
        except (RuntimeError, ValueError):
            return None
        return output.size()
