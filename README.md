# VIAM PYTORCH ML MODEL 

This is a [Viam module](https://docs.viam.com/extend/modular-resources/) providing a mlmodel service for PyTorch model

## Getting started


## Installation with `pip install` 

```
pip install -r requirements.txt
```

## Configure your `mlmodel:torch-cpu` vision service

> [!NOTE]  
> Before configuring your vision service, you must [create a robot](https://docs.viam.com/manage/fleet/robots/#add-a-new-robot).

Navigate to the **Config** tab of your robot’s page in [the Viam app](https://app.viam.com/). Click on the **Services** subtab and click **Create service**. Select the `MLModel` type, then select the `torch-cpi` model. Enter a name for your service and click **Create**.

### Example


```json
{
    "modules": [
     {
      "name": "mymodel",
      "version": "latest",
      "type": "registry",
      "module_id": "viam:torch-cpu"
    }
  ],
  "services": [
    {
      "name": "torch",
      "type": "mlmodel",
      "model": "viam:mlmodel:torch-cpu",
      "attributes": {
        "model_path": "examples/resnet_18/resnet-18.pt", 
        "label_path": "examples/resnet_18/labels.txt", 
      }
    }
  ]
}
```


### Attributes description

The following attributes are available to configure your module:


| Name         | Type   | Inclusion    | Default | Description                       |
| ------------ | ------ | ------------ | ------- | --------------------------------- |
| `model_path` | string | **Required** |         | Path to **standalone** model file |
| `label_path` | string | Optional     |         | Path to file with class labels.   |




# Methods
## `infer()`
```
infer(input_tensors: Dict[str, NDArray], *, timeout: Optional[float]) -> Dict[str, NDArray]
```

### Example

```python
my_model = MLModelClient.from_robot(robot, "torch")
input_image = np.array(Image.open(path_to_input_image), dtype=np.float32)
input_image = np.transpose(input_image, (2, 0, 1))  # channel first
input_image = np.expand_dims(input_image, axis=0)  # batch dim
input_tensor = dict()
input_tensor["input"] = input_image
output = await my_model.infer(input_tensor)
print(f"output.shape is {output['output'].shape}")
```
