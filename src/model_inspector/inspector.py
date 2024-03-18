import torch.nn as nn
import torch
from .input_size_calculator import InputSizeCalculator
from .input_tester import InputTester
from viam.services.mlmodel import MLModel, Metadata, TensorInfo
from viam.logging import getLogger

LOGGER = getLogger(__name__)


class Inspector:
    def __init__(self, model: nn.Module) -> None:
        # self.summary: ModelStatistics = summary(module, input_size=[1,3, 640,480])
        self.model = model
        self.input_size_calculator = InputSizeCalculator()

        self.dimensionality = None

    def find_metadata(self):
        input_shape_candidate = self.reverse_module()
        self.input_tester = InputTester(self.model, input_shape_candidate)
        input_shape, output_shape = self.input_tester.get_shapes()

        if output_shape is not None:
            LOGGER.info(
                f"found input shape: {input_shape} and output shape {output_shape}"
            )
            return (input_shape, output_shape)

    def reverse_module(self):
        modules = list(self.model.children())
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
                continue  # sometimes some modules are None

        return input_shape

    def is_valid_input_shape(self, input_shape):
        input_tensor = torch.ones(input_shape)
        try:
            output = self.model(input_tensor)
        except RuntimeError:
            return None
        except ValueError:
            return None
        return output.size()
