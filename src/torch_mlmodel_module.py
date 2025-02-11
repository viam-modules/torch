"""
This module provides functionality to infer input size predictions 
and retrieve metadata associated with a model.

The module initializes by loading a TorchModel from a specified model file path and 
configures an Inspector to extract metadata, including labels if provided.
"""
from typing import ClassVar, Mapping, Sequence, Dict, Optional
from numpy.typing import NDArray
from typing_extensions import Self
from viam.services.mlmodel import MLModel, Metadata
from viam.module.types import Reconfigurable
from viam.resource.types import Model, ModelFamily
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.utils import ValueTypes
from viam.logging import getLogger
from model.model import TorchModel
from model_inspector.inspector import Inspector

LOGGER = getLogger(__name__)


class TorchMLModelModule(MLModel, Reconfigurable):
    """
    This class integrates a PyTorch model with Viam's MLModel and Reconfigurable interfaces,
    providing functionality to create, configure, and use the model for inference.
    """

    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "mlmodel"), "torch-cpu")

    def __init__(self, name: str):
        super().__init__(name=name)
        self.path_to_model_file = None

        self.torch_model = None
        self.inspector = None
        self._metadata = None

    @classmethod
    def new_service(
        cls, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        "Create and configure a new instance of the service."
        service = cls(config.name)
        service.reconfigure(config, dependencies)
        return service

    @classmethod
    def validate_config(cls, config: ServiceConfig) -> Sequence[str]:
        "Validate the configuration for the service."
        model_path = config.attributes.fields["model_path"].string_value
        if model_path == "":
            raise Exception(
                "model_path can't be empty. model is required for torch mlmoded service module."
            )
        return []

    def reconfigure(
        self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        "Reconfigure the service with the given configuration and dependencies."

        # pylint: disable=too-many-return-statements
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
            if type_default == int:
                return int(config.attributes.fields[attribute_name].number_value)
            if type_default == float:
                return config.attributes.fields[attribute_name].number_value
            if type_default == str:
                return config.attributes.fields[attribute_name].string_value
            if type_default == dict:
                return dict(config.attributes.fields[attribute_name].struct_value)
            return default

        self.path_to_model_file = get_attribute_from_config("model_path", None, str)
        label_file = get_attribute_from_config("label_path", None, str)

        self.torch_model = TorchModel(path_to_serialized_file=self.path_to_model_file)
        self.inspector = Inspector(self.torch_model)
        self._metadata = self.inspector.find_metadata(label_file)

    async def infer(
        self,
        input_tensors: Dict[str, NDArray],
        *,
        extra: Optional[Mapping[str, ValueTypes]],
        timeout: Optional[float],
    ) -> Dict[str, NDArray]:
        """Take an already ordered input tensor as an array,
        make an inference on the model, and return an output tensor map.

        Args:
            input_tensors (Dict[str, NDArray]):
                A dictionary of input flat tensors as specified in the metadata

        Returns:
            Dict[str, NDArray]:
                A dictionary of output flat tensors as specified in the metadata
        """
        return self.torch_model.infer(input_tensors)

    async def metadata(
            self,
            *,
            extra: Optional[Mapping[str, ValueTypes]],
            timeout: Optional[float],
    ) -> Metadata:
        """Get the metadata (such as name, type, expected tensor/array shape,
        inputs, and outputs) associated with the ML model.

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
