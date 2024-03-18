from torch.nn import Module, Linear
from typing import Tuple
import torch


def is_valid_input_shape(model, input_shape, add_batch_dimension: bool = False):
    """
    Check if the input shape is valid for a given PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to validate the input shape for.
        input_shape (tuple): The shape of the input tensor. It should be in the format (C, H, W) for image-like data,
                             where C is the number of channels, H is the height, and W is the width.
        add_batch_dimension (bool, optional): Whether to add a batch dimension to the input tensor. Default is False.

    Returns:
        list or None: A list representing the shape of the output tensor if the input shape is valid for the model,
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
    return list(output.size())


def is_defined_shape(shape: Tuple[int]):
    if shape is None:
        return False
    return -1 not in shape


def inspect_layer(layer: Module):
    if isinstance(layer, Linear):
        return {
            "dimensionality": 2,
            "input_shape": [1, layer.in_features],
            "output_shape": [1, layer.out_features],
        }

    dim = 0
    return dim


def map_layer_type_and_dimensionnality(layer: Module):
    if isinstance(layer, Linear):
        return 1


def dimensionality_unicity(sizes: list):
    if not sizes:
        return None

    dimensionality = len(sizes[0])
    if not all(len(size) == dimensionality for size in sizes):
        return None
    return dimensionality


def solve_shape(sizes, dimensionality):
    """
    Determine the common size for each dimension among given sizes.

    Args:
    - sizes (list): List of tuples representing sizes in each dimension.
    - dimensionality (int): Number of dimensions.

    Returns:
    - list: List containing the common size for each dimension. -1 indicates 'any'.
    """
    if not sizes or not sizes[0]:
        return [-1] * dimensionality

    result = list(sizes[0])
    for dim_index in range(dimensionality):
        common_size = sizes[0][dim_index]
        for size in sizes:
            if size[dim_index] != common_size:
                result[dim_index] = -1
                break
    return result
