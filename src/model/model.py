import torch
from typing import List, Iterable, Dict, Any
from numpy.typing import NDArray
import torch.nn as nn
from collections import OrderedDict
from viam.logging import getLogger

LOGGER = getLogger(__name__)


class TorchModel:
    def __init__(
        self,
        path_to_serialized_file: str,
        model: nn.Module = None,
    ) -> None:
        if model is not None:
            self.model = model
        else:
            self.model = torch.load(path_to_serialized_file)
        if not isinstance(self.model, nn.Module):
            if isinstance(self.model, OrderedDict):
                LOGGER.error(
                    f"the file {path_to_serialized_file} provided as model file is of type collections.OrderedDict, which suggests that the provided file describes weights instead of a standalone model"
                )
            raise TypeError(
                f"the model is of type {type(self.model)} instead of nn.Module type"
            )
        self.model.eval()

    def infer(self, input):
        input = self.prepare_input(input)
        with torch.no_grad():
            output = self.model(*input)
        return self.wrap_output(output)

    @staticmethod
    def prepare_input(input_tensor: Dict[str, NDArray]) -> List[NDArray]:
        return [torch.from_numpy(tensor) for tensor in input_tensor.values()]

    @staticmethod
    def wrap_output(output: Any) -> Dict[str, NDArray]:
        if isinstance(output, Iterable):
            if len(output) == 1:
                output = output[0]  # unpack batched results

        if isinstance(output, torch.Tensor):
            return {"output_0": output.numpy()}

        elif isinstance(output, dict):
            for tensor_name, tensor in output.items():
                if isinstance(tensor, torch.Tensor):
                    output[tensor_name] = tensor.numpy()

            return output
        elif isinstance(output, Iterable):
            res = {}
            count = 0
            for out in output:
                res[f"output_{count}"] = out
                count += 1
            return res

        else:
            raise TypeError(f"can't convert output of type {type(output)} to array")
