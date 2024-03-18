import torch.nn as nn
import torch
from torchinfo import summary, ModelStatistics
from torch.jit import ScriptModule
from utils import dimensionality_unicity, solve_shape, is_defined_shape
from input_size_calculator import InputSizeCalculator
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
        input_shape = self.find_input_shape()
        output_shape = self.is_valid_input_shape(input_shape)
        if output_shape is not None:
            LOGGER.info(
                f"found input shape: {input_shape} and output shape {output_shape}"
            )
            return (input_shape, output_shape)

    def find_input_shape(self):
        if isinstance(self.model, nn.Module):
            input_shape = self.reverse_module()
            if input_shape is not None:
                return input_shape

        else:
            input_shape = self.try_random_image_input()
            if input_shape is not None:
                return input_shape
            else:
                return None

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

    def try_random_image_input(self, n_dims=None) -> TensorInfo:
        working_sizes = []
        not_working_sizes = []

        rgb_size_batched_1 = torch.Size([1, 3, 224, 224])
        rgb_size_batched_2 = torch.Size([1, 3, 112, 112])

        grey_size_batched_1 = torch.Size([1, 1, 224, 224])
        grey_size_batched_2 = torch.Size([1, 1, 112, 112])

        rgb_size_1 = torch.Size([3, 224, 224])
        rgb_size_2 = torch.Size([3, 112, 112])

        grey_size_1 = torch.Size([1, 224, 224])
        grey_size_2 = torch.Size([1, 112, 112])

        sizes = [
            rgb_size_batched_1,
            rgb_size_batched_2,
            grey_size_batched_1,
            grey_size_batched_2,
            rgb_size_1,
            rgb_size_2,
            grey_size_1,
            grey_size_2,
        ]

        for size in sizes:
            random_t = torch.rand(size)
            try:
                self.model(random_t)
            except RuntimeError:
                not_working_sizes.append(size)
            except ValueError:
                not_working_sizes.append(size)
            else:
                working_sizes.append(size)
                print(f"Working dimensionnality : {len(size)}")

        # self.working_sizes= working_sizes
        # self.not_working_sizes = not_working_sizes
        dimensionnality = dimensionality_unicity(
            working_sizes
        )  # if None here it's over
        if dimensionnality is not None:
            shape = solve_shape(working_sizes, dimensionnality)
        # elif isinstance(self.module, Sequential): #we can try to reverse it

        return shape
