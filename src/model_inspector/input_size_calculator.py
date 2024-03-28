from .utils import is_defined_shape
import torch.nn as nn
from typing import Dict, Tuple
from viam.logging import getLogger

LOGGER = getLogger(__name__)


class InputSizeCalculator:
    """

    Given a layer returns the input size.
    There are two types of layers. Those whose input shape is known without knowing
    the output shape, and those whose input shape can't be known without knowing the output shape.
    From, the former we still extract information about dimensionnality.


    Note:
        Everything unbatched

    Usage:
         output_shape = InputSizeCalculator.get_input_size(
                    layer, input_shape
                )

    Args:


    """

    @staticmethod
    def linear(
        layer: nn.Linear, output_shape: Tuple[int], return_all: bool = False
    ) -> Tuple[int]:
        return (
            1,
            layer.in_features,
        )

    @staticmethod
    def rnn(
        layer: nn.RNN, output_shape: Tuple[int], return_all: bool = False
    ) -> Tuple[int]:
        H_in = layer.input_size
        if output_shape is None:
            L = -1
        elif layer.batch_first:
            L = output_shape[1]
        else:
            L = output_shape[0]
        return (L, H_in)

    @staticmethod
    def lstm(
        layer: nn.LSTM, output_shape: Tuple[int], return_all: bool = False
    ) -> Tuple[int]:
        H_in = layer.input_size
        if output_shape is None:
            L = -1
        elif layer.batch_first:
            L = output_shape[1]
        else:
            L = output_shape[0]
        return (L, H_in)

    @staticmethod
    def embedding(
        layer: nn.Embedding, output_shape: Tuple[int], return_all: bool = False
    ) -> Tuple[int]:
        if output_shape is None:
            return -1
        return output_shape[0]

    @staticmethod
    def layer_norm(
        layer: nn.LayerNorm, output_shape: Tuple[int], return_all: bool = False
    ) -> Tuple[int]:
        return layer.normalized_shape

    @staticmethod
    def batch_norm_1d(
        layer: nn.BatchNorm1d, output_shape: Tuple[int], return_all: bool = False
    ) -> Tuple[int]:
        C = layer.num_features
        if output_shape is None:
            L = -1
        else:
            L = output_shape[-1]
        return (C, L)

    @staticmethod
    def batch_norm_2d(
        layer: nn.BatchNorm2d, output_shape: Tuple[int], return_all: bool = False
    ) -> Tuple[int]:
        C = layer.num_features
        if output_shape is None:
            H, W = -1, -1
        else:
            H, W = output_shape[-2], output_shape[-1]
        return (C, H, W)

    @staticmethod
    def batch_norm_3d(
        layer: nn.BatchNorm3d, output_shape: Tuple[int], return_all: bool = False
    ) -> Tuple[int]:
        # TODO
        return output_shape

    @staticmethod
    def maxpool_1d(
        layer: nn.MaxPool1d, output_shape: Tuple[int], return_all: bool = False
    ) -> Tuple[int]:
        if output_shape is None or not is_defined_shape(output_shape):
            return (-1, -1)

        C, L_out = output_shape

        padding = layer.padding
        dilation = layer.dilation
        ks = layer.kernel_size
        stride = layer.stride
        L_in = stride * (L_out - 1) - 2 * padding + dilation * (ks - 1) + 1
        if return_all:
            res = []
            for i in range(stride):
                res.append((C, L_in + i))
            return res

        else:
            return (C, L_in)

    @staticmethod
    def maxpool_2d(
        layer: nn.MaxPool2d, output_shape: Tuple[int], return_all: bool = False
    ) -> Tuple[int]:
        if output_shape is None or not is_defined_shape(output_shape):
            return (-1, -1, -1)  # (C, H, W)
        C_out, H_out, W_out = output_shape
        padding = layer.padding
        dilation = layer.dilation
        ks = layer.kernel_size
        stride = layer.stride

        padding = layer.padding
        dilation = layer.dilation
        ks = layer.kernel_size
        stride = layer.stride

        H_in = stride[0] * (H_out - 1) - 2 * padding[0] + dilation[0] * (ks[0] - 1) + 1
        W_in = stride[1] * (W_out - 1) - 2 * padding[1] + dilation[1] * (ks[1] - 1) + 1

        if return_all:
            res = []
            for i in range(stride[0]):
                for j in range(stride[1]):
                    res.append((C_out, H_in + i, W_in + j))

            return res
        else:
            return (C_out, H_in, W_in)

    @staticmethod
    def avgpool_1d(
        layer: nn.AvgPool1d, output_shape: Tuple[int], return_all: bool = False
    ) -> Tuple[int]:
        if output_shape is None or not is_defined_shape(output_shape):
            return (-1, -1)

        C, L_out = output_shape

        padding = layer.padding
        ks = layer.kernel_size
        stride = layer.stride
        L_in = stride * (L_out - 1) - 2 * padding + ks
        if return_all:
            res = []
            for i in range(stride):
                res.append((C, L_in + i))
            return res
        else:
            return (C, L_in)

    @staticmethod
    def avgpool_2d(
        layer: nn.AvgPool2d, output_shape: Tuple[int], return_all: bool = False
    ) -> Tuple[int]:
        if output_shape is None or not is_defined_shape(output_shape):
            return (-1, -1, -1)
        C, L_out = output_shape

        padding = layer.padding
        ks = layer.kernel_size
        stride = layer.stride
        L_in = stride * (L_out - 1) - 2 * padding + ks
        if return_all:
            res = []
            for i in range(stride):
                res.append((C, L_in + i))
            return res
        else:
            return (C, L_in)

    @staticmethod
    def conv_1d(
        layer: nn.Conv1d, output_shape: Tuple[int], return_all: bool = False
    ) -> Tuple[int]:
        if output_shape is None or not is_defined_shape(output_shape):
            return (layer.in_channels, -1)
        C, L_out = output_shape

        padding = layer.padding
        dilation = layer.dilation
        ks = layer.kernel_size
        stride = layer.stride
        L_in = stride * (L_out - 1) - 2 * padding + dilation * (ks - 1) + 1
        res = []
        for i in range(stride):
            res.append((C, L_in + i))
        return res

    @staticmethod
    def conv_2d(
        layer: nn.Conv2d, output_shape: Tuple[int], return_all: bool = False
    ) -> Tuple[int]:
        if output_shape is None or not is_defined_shape(output_shape):
            return (layer.in_channels, -1, -1)
        H_out, W_out = output_shape[-2], output_shape[-1]

        padding = layer.padding
        dilation = layer.dilation
        ks = layer.kernel_size
        stride = layer.stride

        H_in = stride[0] * (H_out - 1) - 2 * padding[0] + dilation[0] * (ks[0] - 1) + 1
        W_in = stride[1] * (W_out - 1) - 2 * padding[1] + dilation[1] * (ks[1] - 1) + 1

        res = []
        if return_all:
            for i in range(stride[0]):
                for j in range(stride[1]):
                    res.append((layer.in_channels, H_in + i, W_in + j))
            return res
        else:
            return (layer.in_channels, H_in, W_in)

    @staticmethod
    def default(layer, output_shape, return_all: bool = False):
        """
        Since this serves a best effort to find input shape for model,
        we assume that layers of unknown types have the same input and output
        shapes.
        The input shape then should be tested.

        Args:
            layer (_type_): _description_
            output_shape (_type_): _description_
            return_all (bool, optional): _description_. Defaults to False.

        """
        # if output_shape is None:
        #     raise NotImplementedError(
        #         f"Input size calculation for {type(layer).__name__} is not implemented"
        #     )
        # else:
        #     return output_shape
        return output_shape

    def __init__(self):
        self.calculators: Dict[type, callable] = {
            # nn.Sequential: self.get_input_size,
            nn.Linear: self.linear,
            nn.RNN: self.rnn,
            nn.Embedding: self.embedding,
            nn.LayerNorm: self.layer_norm,
            nn.BatchNorm1d: self.batch_norm_1d,
            nn.BatchNorm2d: self.batch_norm_2d,
            nn.BatchNorm3d: self.batch_norm_3d,
            nn.MaxPool1d: self.maxpool_1d,
            nn.MaxPool2d: self.maxpool_2d,
            nn.Conv1d: self.conv_1d,
            nn.Conv2d: self.conv_2d,
        }

    def get_input_size(
        self, layer: nn.Module, output_shape: Tuple[int], return_all: bool = False
    ) -> Tuple[int]:
        layer_type = type(layer)
        calculator = self.calculators.get(layer_type, self.default)

        input_shape = calculator(layer, output_shape, return_all)
        LOGGER.info(
            f" For layer type {type(layer).__name__}, with output shape: {output_shape} input shape found is {input_shape}"
        )
        return input_shape
