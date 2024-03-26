from typing import ClassVar, Mapping, Sequence, Dict, Optional
from numpy.typing import NDArray
from typing_extensions import Self
from viam.services.mlmodel import MLModel, Metadata, TensorInfo
from viam.module.types import Reconfigurable
from viam.resource.types import Model, ModelFamily
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.utils import ValueTypes
from viam.logging import getLogger
from .model.model import TorchModel
from .model_inspector.inspector import Inspector
import torch

LOGGER = getLogger(__name__)


class TorchMLModelModule(MLModel, Reconfigurable):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "mlmodel"), "torch")

    def __init__(self, name: str):
        super().__init__(name=name)

    @classmethod
    def new_service(
        cls, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        service = cls(config.name)
        service.reconfigure(config, dependencies)
        return service

    @classmethod
    def validate_config(cls, config: ServiceConfig) -> Sequence[str]:
        serialized_file = config.attributes.fields["model_file"].string_value
        if serialized_file == "":
            raise Exception(
                "model_file can't be empty. model is required for torch mlmoded service module."
            )
        return []

    def reconfigure(
        self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        def get_attribute_from_config(attribute_name: str, default, of_type=None):
            if attribute_name not in config.attributes.fields:
                return default

            if default is None:
                if of_type is None:
                    raise Exception(
                        "If default value is None, of_type argument can't be empty"
                    )
                type_default = of_type
            else:
                type_default = type(default)
            if type_default == bool:
                return config.attributes.fields[attribute_name].bool_value
            elif type_default == int:
                return int(config.attributes.fields[attribute_name].number_value)
            elif type_default == float:
                return config.attributes.fields[attribute_name].number_value
            elif type_default == str:
                return config.attributes.fields[attribute_name].string_value
            elif type_default == dict:
                return dict(config.attributes.fields[attribute_name].struct_value)

        # TODO: Test self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.path_to_model_file = get_attribute_from_config("model_file", None, str)
        self.path_to_label_file = get_attribute_from_config("label_file", None, str)
        self.model_type = get_attribute_from_config("model_type", None, str)
        label_file = get_attribute_from_config("label_path", None, str)

        self.torch_model = TorchModel(path_to_serialized_file=self.path_to_model_file)
        self.inspector = Inspector(self.torch_model.model)
        self._metadata = self.inspector.find_metadata(label_file)
        self.input_names = ["input"]
        self.output_names = ["output"]

    async def infer(
        self, input_tensors: Dict[str, NDArray], *, timeout: Optional[float]
    ) -> Dict[str, NDArray]:
        """Take an already ordered input tensor as an array, make an inference on the model, and return an output tensor map.

        Args:
            input_tensors (Dict[str, NDArray]): A dictionary of input flat tensors as specified in the metadata

        Returns:
            Dict[str, NDArray]: A dictionary of output flat tensors as specified in the metadata
        """
        return self.torch_model.infer(input_tensors)

    async def metadata(self, *, timeout: Optional[float]) -> Metadata:
        """Get the metadata (such as name, type, expected tensor/array shape, inputs, and outputs) associated with the ML model.

        Returns:
            Metadata: The metadata
        """

        return self._metadata

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        raise NotImplementedError
