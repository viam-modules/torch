from typing import List, Optional, Dict
from model_inspector.utils import is_defined_shape, output_to_shape_dict
import torch


class InputTester:
    def __init__(self, model, input_candidate=None):
        """
        A class for testing input shapes on a PyTorch model.

        This class provides methods to test various input shapes on a given PyTorch model and collect information
        about working and non-working input sizes.

        Args:
            model (torch.nn.Module): The PyTorch model to be tested.

        Note:
            The try_image_input and try_audio_input methods test the model with predefined input sizes for image-like and
            audio-like data, respectively. The get_shapes method retrieves the final input and output shapes after testing
            various input sizes.
        """
        self.model = model
        self.input_candidate = input_candidate
        self.working_input_sizes = {
            "input": []
        }  # input tester only test single tensor inputs
        self.working_output_sizes: Dict[
            str, List[List[int]]
        ] = {}  # eg: {output_0 = [[1,2], [1,2]], output_1 = [[2], [2]]}

    @staticmethod
    def dimensionality_unicity(sizes: List[List[int]]) -> bool:
        """
        Check if the dimensionality of the input sizes is unique.

        Args:
            sizes (List[List[int]]): List of input sizes.

        Returns:
           bool: True if dimensionality is unique among sizes. Else, False.
        """
        if not sizes:
            return False
        dimensionality_set = {len(size) for size in sizes}
        if len(dimensionality_set) == 1:
            return True
        return False

    @staticmethod
    def solve_shape(sizes: List[List[int]]):
        """
        Solve the shape of the input sizes.

        Args:
            sizes (List[List[int]]): List of input sizes.

        Returns:
            List[int]: The solved shape.
        """
        res = sizes[0]
        for i in range(len(sizes[0])):
            for size in sizes[1:]:
                if res[i] != size[i]:
                    res[i] = -1
        return res

    def try_image_input(self, n_dims: Optional[int] = None):
        """
        Try some image-like inputs
        """
        rgb_size_1 = [3, 224, 224]
        rgb_size_2 = [3, 112, 112]
        # TODO: add 'weirder' size like [3,113, 217]

        grey_size_1 = [1, 224, 224]
        grey_size_2 = [1, 112, 112]

        input_sizes = [
            rgb_size_1,
            rgb_size_2,
            grey_size_1,
            grey_size_2,
        ]

        input_sizes_batched = [[1] + array for array in input_sizes]
        input_sizes += input_sizes_batched

        for input_size in input_sizes:
            self.test_input_size(input_size)

    def try_audio_input(self):
        """
        Try some audio-like inputs
        """
        sampling_rates = [8e3, 16e3, 44.1e3]  # 8 kHz, 16 kHz, 44.1 kHz

        one_second_mono = [[1, int(sr)] for sr in sampling_rates]
        one_second_stereo = [[2, int(sr)] for sr in sampling_rates]
        five_second_mono = [[1, int(5 * sr)] for sr in sampling_rates]
        five_second_stereo = [[2, int(5 * sr)] for sr in sampling_rates]

        input_sizes = (
            one_second_mono + one_second_stereo + five_second_mono + five_second_stereo
        )

        input_sizes_batched = [[1] + array for array in input_sizes]
        input_sizes += input_sizes_batched

        for input_size in input_sizes:
            self.test_input_size(input_size)

    def test_input_size(self, input_size):
        input_array = torch.ones(
            (input_size)
        ).numpy()  # i get type issues when using np.ones()
        input_tensor = {"input": input_array}
        output = None
        try:
            output = self.model.infer(input_tensor)
        except (RuntimeError, ValueError, AssertionError):
            pass
        if output is not None:
            self.working_input_sizes["input"].append(input_size)

            outputs_shape = output_to_shape_dict(output)
            for output, shape in outputs_shape.items():
                if output in self.working_output_sizes:
                    self.working_output_sizes[output].append(shape)
                else:
                    self.working_output_sizes[output] = [shape]

    def try_inputs(self):
        if self.input_candidate:
            if is_defined_shape(self.input_candidate):
                self.test_input_size(self.input_candidate)
        self.try_image_input()
        self.try_audio_input()

    def get_shapes(self):
        self.try_inputs()
        input_shapes, output_shapes = {}, {}
        for output_tensor_name, sizes in self.working_output_sizes.items():
            if self.dimensionality_unicity(sizes):
                output_shapes[output_tensor_name] = self.solve_shape(sizes)
            else:
                raise Exception(
                    f"dimensionality for valid output: {output_tensor_name} is not unique"
                )

        for input_tensor_name, sizes in self.working_input_sizes.items():
            if self.dimensionality_unicity(sizes):
                input_shapes[input_tensor_name] = self.solve_shape(sizes)
            else:
                raise Exception(
                    f"dimensionality for valid input: {input_tensor_name} is not unique"
                )
        return input_shapes, output_shapes
