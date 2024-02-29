# VIAM PYTORCH ML MODEL 
***in progress***

This is a [Viam module](https://docs.viam.com/extend/modular-resources/) providing a mlmodel service for PyTorch model

## Getting started


## Installation with `pip install` 

```
pip install -r requirements.txt
```

## Configure your `mlmodel:torch` vision service

> [!NOTE]  
> Before configuring your vision service, you must [create a robot](https://docs.viam.com/manage/fleet/robots/#add-a-new-robot).

Navigate to the **Config** tab of your robot’s page in [the Viam app](https://app.viam.com/). Click on the **Services** subtab and click **Create service**. Select the `Vision` type, then select the `deepface_identification` model. Enter a name for your service and click **Create**.

### Example


```json
{
    "modules": [
    {
      "executable_path": "/Users/robinin/torch-infer/torch/run.sh",
      "name": "mymodel",
      "type": "local"
    }
  ],
  "services": [
    {
      "name": "torch",
      "type": "mlmodel",
      "model": "viam:mlmodel:torch",
      "attributes": {
        "path_to_serialized_file": "/Users/robinin/torch-infer/torch/resnet_18/resnet-18.pt", 
				"model_type": "vision"
      }
    }
  ]
}
```


### Attributes description

The following attributes are available to configure your deepface module:


| Name                       | Type   | Inclusion    | Default  | Description                                                                                  |
| -------------------------- | ------ | ------------ | -------  | -------------------------------------------------------------------------------------------- |
| `path_to_serialized_file`  | string | **Required** |          | FOR NOW: this can only be a TorchScript model                                                |
| `model_type`               | string | Optional     | `'yunet'`| FOR NOW: `object_detector` or `vision`. This is used for preprocessing.                      |



## IR input support



