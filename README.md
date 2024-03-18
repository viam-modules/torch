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
| `model_type`               | string | Optional     | `None`| FOR NOW: `object_detector` or `vision`. This is used for preprocessing.                      |




### Supported size defining layers
# Reverting a nn.Sequential:

### Supported size defining layers

def: The input size is directly checkable or easy to find given the attributes of the layer

1. **Linear Layer (Fully Connected Layer)**:
    - **`torch.nn.Linear(in_features, out_features, bias=True)`**
2. **RNN (Recurrent Neural Network) Layers**:
    - **`torch.nn.RNN(input_size, hidden_size, num_layers, ...)`**
    - **`torch.nn.LSTM(input_size, hidden_size, num_layers, ...)`**
    - **`torch.nn.GRU(input_size, hidden_size, num_layers, ...)`**
3. **MLP (Multi-Layer Perceptron) Layers**:
    - **`torch.nn.Sequential(*args)`**
4. **Embedding Layer**:
    - **`torch.nn.Embedding(num_embeddings, embedding_dim)`**
5. **Transformer Layers**:
    - **`torch.nn.TransformerEncoderLayer(d_model, nhead)`**
    - **`torch.nn.TransformerDecoderLayer(d_model, nhead)`**
6. **Normalization Layers**:
    - **`torch.nn.LayerNorm(normalized_shape, ...)`**
    - **`torch.nn.BatchNorm1d(num_features, ...)`**
    - **`torch.nn.BatchNorm2d(num_features, ...)`**
    - **`torch.nn.BatchNorm3d(num_features, ...)`**
7. **Pooling Layers**:
    - **`torch.nn.MaxPool1d(...)`**
    - **`torch.nn.MaxPool2d(...)`**
    - **`torch.nn.MaxPool3d(...)`**
    - **`torch.nn.AvgPool1d(...)`**
    - **`torch.nn.AvgPool2d(...)`**
    - **`torch.nn.AvgPool3d(...)`**
8. **Flattening Layer**:

### Supported invertible layers

 def: given the output size you can guess the input