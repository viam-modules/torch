# VIAM PYTORCH ML MODEL 
***in progress***

This is a [Viam module](https://docs.viam.com/extend/modular-resources/) providing a mlmodel service for PyTorch model

## Getting started


## Installation with `pip install` 

```
pip install -r requirements.txt
```

## Configure your `mlmodel:torch-cpu` vision service

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


| Name         | Type   | Inclusion    | Default | Description                                                                                          |
| ------------ | ------ | ------------ | ------- | ---------------------------------------------------------------------------------------------------- |
| `model_path` | string | **Required** |         | Path to **standalone** model file or [torchvision model](#avalaible-model-names-on-torchvision0162). |
| `label_path` | string | Optional     |         | Path to file with class labels.                                                                      |




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



## Torchvision model

You can load a pretrained model from torchvision model zoo. You can use the prefix  `torchvision://` followed by the model name. The MLModel service will use [Torchvision Multi-Weight API](https://pytorch.org/blog/introducing-torchvision-new-multi-weight-support-api/) to load model architecture and `DEFAULT` weigths.

### Example


```json
{
  "services": [
    {
      "name": "torch",
      "type": "mlmodel",
      "model": "viam:mlmodel:torch-cpu",
      "attributes": {
        "model_path": "torchvision://resnet18", 
      }
    }
  ]
}
```


### Avalaible model names on `torchvision@0.16.2`
Available models are the one returned by the torchvision API call to `list_models()`.

```
['alexnet', 'convnext_base', 'convnext_large', 'convnext_small', 'convnext_tiny', 'deeplabv3_mobilenet_v3_large', 'deeplabv3_resnet101', 'deeplabv3_resnet50', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_l', 'efficientnet_v2_m', 'efficientnet_v2_s', 'fasterrcnn_mobilenet_v3_large_320_fpn', 'fasterrcnn_mobilenet_v3_large_fpn', 'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2', 'fcn_resnet101', 'fcn_resnet50', 'fcos_resnet50_fpn', 'googlenet', 'inception_v3', 'keypointrcnn_resnet50_fpn', 'lraspp_mobilenet_v3_large', 'maskrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn_v2', 'maxvit_t', 'mc3_18', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'mvit_v1_b', 'mvit_v2_s', 'quantized_googlenet', 'quantized_inception_v3', 'quantized_mobilenet_v2', 'quantized_mobilenet_v3_large', 'quantized_resnet18', 'quantized_resnet50', 'quantized_resnext101_32x8d', 'quantized_resnext101_64x4d', 'quantized_shufflenet_v2_x0_5', 'quantized_shufflenet_v2_x1_0', 'quantized_shufflenet_v2_x1_5', 'quantized_shufflenet_v2_x2_0', 'r2plus1d_18', 'r3d_18', 'raft_large', 'raft_small', 'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf', 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_128gf', 'regnet_y_16gf', 'regnet_y_1_6gf', 'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext101_64x4d', 'resnext50_32x4d', 'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2', 's3d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'ssd300_vgg16', 'ssdlite320_mobilenet_v3_large', 'swin3d_b', 'swin3d_s', 'swin3d_t', 'swin_b', 'swin_s', 'swin_t', 'swin_v2_b', 'swin_v2_s', 'swin_v2_t', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'vit_b_16', 'vit_b_32', 'vit_h_14', 'vit_l_16', 'vit_l_32', 'wide_resnet101_2', 'wide_resnet50_2']
```
 **Warning:** This doesn't guarantee that the weights associated to the model will work.