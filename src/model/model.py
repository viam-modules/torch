"""
This module provides a class for loading and performing inference with a PyTorch model.
The TorchModel class handles loading a serialized model, preparing inputs, and wrapping outputs.
"""

import os
from typing import List, Iterable, Dict, Any
from collections import OrderedDict

from numpy.typing import NDArray
from viam.logging import getLogger

import torch
from torch import nn


LOGGER = getLogger(__name__)


class TorchModel:
    """
    A class to load a PyTorch model from a serialized file or use a provided model,
    prepare inputs for the model, perform inference, and wrap the outputs.
    """

    def __init__(
        self,
        path_to_serialized_file: str,
        model: nn.Module = None,
    ) -> None:
        "Initializes the model by loading it from a serialized file or using a provided model."
        if model is not None:
            self.model = model
        else:
            size_mb = os.stat(path_to_serialized_file).st_size / (1024 * 1024)
            if size_mb > 500:
                # pylint: disable=deprecated-method
                LOGGER.warn(
                    "model file may be large for certain hardware (%s MB)", size_mb
                )
            self.model = torch.load(path_to_serialized_file)
        if not isinstance(self.model, nn.Module):
            if isinstance(self.model, OrderedDict):
                LOGGER.error(
                    """the file %s provided as model file 
                    is of type collections.OrderedDict, 
                    which suggests that the provided file 
                    describes weights instead of a standalone model""",
                    path_to_serialized_file,
                )
            raise TypeError(
                f"the model is of type {type(self.model)} instead of nn.Module type"
            )
        self.model.eval()

    def infer(self, input_data):
        "Prepares the input data, performs inference using the model, and wraps the output."
        input_data = self.prepare_input(input_data)
        with torch.no_grad():
            output = self.model(*input_data)
        return self.wrap_output(output)

    @staticmethod
    def prepare_input(input_tensor: Dict[str, NDArray]) -> List[NDArray]:
        "Converts a dictionary of NumPy arrays into a list of PyTorch tensors."
        return [torch.from_numpy(tensor) for tensor in input_tensor.values()]

    @staticmethod
    def wrap_output(output: Any) -> Dict[str, NDArray]:
        "Converts the output from a PyTorch model to a dictionary of NumPy arrays."
        if isinstance(output, Iterable):
            if len(output) == 1:
                output = output[0]  # unpack batched results

        if isinstance(output, torch.Tensor):
            return {"output_0": output.numpy()}

        if isinstance(output, dict):
            for tensor_name, tensor in output.items():
                if isinstance(tensor, torch.Tensor):
                    output[tensor_name] = tensor.numpy()

            return output

        if isinstance(output, Iterable):
            res = {}
            count = 0
            for out in output:
                res[f"output_{count}"] = out
                count += 1
            return res

        raise TypeError(f"can't convert output of type {type(output)} to array")
