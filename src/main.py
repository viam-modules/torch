import asyncio


from viam.module.module import Module
from viam.resource.registry import Registry, ResourceCreatorRegistration
from .torch_mlmodel_module import TorchMLModelModule
from viam.services.mlmodel import MLModel


async def main():
    """
    This function creates and starts a new module, after adding all desired
    resource models. Resource creators must be registered to the resource
    registry before the module adds the resource model.
    """
    Registry.register_resource_creator(
        MLModel.SUBTYPE,
        TorchMLModelModule.MODEL,
        ResourceCreatorRegistration(
            TorchMLModelModule.new_service, TorchMLModelModule.validate_config
        ),
    )
    module = Module.from_args()

    module.add_model_from_registry(MLModel.SUBTYPE, TorchMLModelModule.MODEL)
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())
