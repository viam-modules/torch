"Utility functions for model inspection and input shape validation."

from typing import Tuple, Dict, List
from numpy.typing import NDArray
import numpy as np
import torch


def is_valid_input_shape(model, input_shape, add_batch_dimension: bool = False):
    """
    Check if the input shape is valid for a given PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to validate the input shape for.
        input_shape (tuple): The shape of the input tensor.
                            It should be in the format (C, H, W) for image-like data,
                            where C is the number of channels, H is the height, and W is the width.
        add_batch_dimension (bool, optional):
                            Whether to add a batch dimension to the input tensor. Default is False.

    Returns:
        list or None: A list representing the shape of the output tensor
                    if the input shape is valid for the model,
                      or None if an exception occurs during the model evaluation.
    """

    input_tensor = torch.ones(input_shape)
    if add_batch_dimension:
        input_tensor.unsqueeze(0)
    try:
        output = model(input_tensor)
    except RuntimeError:
        return None
    except ValueError:
        return None
    if isinstance(output, torch.Tensor):
        out = list(output.size())
        return out
    return None


def is_defined_shape(shape: Tuple[int]) -> bool:
    """

    Check if a shape input variable defines a shape or
    is just information about dimensionnality.

    Args:
        shape (Tuple[int]):

    Returns:
        _type_: _description_
    """
    if shape is None:
        return False
    return -1 not in shape


def output_to_shape_dict(output: Dict[str, NDArray]) -> Dict[str, List[int]]:
    """takes a model input (Tensor or dict)

    Args:
        output (Dict[str, NDArray]): _description_

    Raises:
        ValueError: _description_

    Returns:
        Dict[str, List[int]]: _description_
    """
    result = {}
    for key, value in output.items():
        if isinstance(value, np.ndarray):
            shape = list(value.shape)
        elif isinstance(value, torch.Tensor):
            shape = list(value.size())
        else:
            raise ValueError(f"can't find shape of type {type(value)}")
        result[key] = shape
    return result
