import torch.nn as nn
import torch
from .input_size_calculator import InputSizeCalculator
from .input_tester import InputTester
from viam.services.mlmodel import MLModel, Metadata, TensorInfo
from viam.logging import getLogger
from viam.utils import dict_to_struct

LOGGER = getLogger(__name__)


class Inspector:
    def __init__(self, model: nn.Module) -> None:
        # self.summary: ModelStatistics = summary(module, input_size=[1,3, 640,480])
        self.model = model
        self.input_size_calculator = InputSizeCalculator()

        self.dimensionality = None

    def find_metadata(self, label_path: str):
        if label_path is not None:
            extra = dict_to_struct({"label": label_path})
        input_info, output_info = [], []
        input_shape_candidate = self.reverse_module()
        self.input_tester = InputTester(self.model, input_shape_candidate)
        input_shapes, output_shapes = self.input_tester.get_shapes()
        for input_tensor_name, shape in input_shapes.items():
            input_info.append(TensorInfo(name=input_tensor_name, shape=shape))

        for output_tensor_name, shape in input_shapes.items():
            output_info.append(
                TensorInfo(name=output_tensor_name, shape=shape, extra=extra)
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

        # TODO: Add output shape from label files
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
                LOGGER.info(f"For module {module}, the output shape is {output_shape}")
            else:
                continue  # sometimes some children are None

        return input_shape

    def is_valid_input_shape(self, input_shape):
        input_tensor = torch.ones(input_shape)
        try:
            output = self.model(input_tensor)
        except (RuntimeError, ValueError):
            return None
        return output.size()
