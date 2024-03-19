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
        "path_to_serialized_file": "examples/resnet_18/resnet-18.pt", 
      }
    }
  ]
}
```


### Attributes description

The following attributes are available to configure your module:


| Name         | Type   | Inclusion    | Default | Description                       |
| ------------ | ------ | ------------ | ------- | --------------------------------- |
| `model_file` | string | **Required** |         | Path to **standalone** model file |


# Methods
## `infer()`
```
infer(input_tensors: Dict[str, NDArray], *, timeout: Optional[float]) -> Dict[str, NDArray]
```

### Input and output dictionnaries.
 - For now, only support single input tensor. Key in the input `Dict` must be `'input'`. 
- For now, only support single output tensor. 
Key in the output `Dict` is `'output'`. 
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

## `metadata()`

