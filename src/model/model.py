import torch
from typing import List, Tuple
from numpy.typing import NDArray
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from viam.logging import getLogger
from torchsummary import summary
LOGGER = getLogger(__name__)

class TorchModel:
    def __init__(self,
                 path_to_serialized_file:str, 
                 path_to_model_file:str=None) -> None:
        self.model = torch.load(path_to_serialized_file)
        if not isinstance(self.model, nn.Module):
            if isinstance(self.model, OrderedDict):
                LOGGER.error(f'the file {path_to_model_file} provided as model file is of type collections.OrderedDict, which suggests that the provided file describes weights instead of a standalone model')
            raise TypeError(f'the model is of type {type(self.model)} instead of nn.Module type')
            
        self.model.eval()
        children =self.model.children()
        print(children)
        
    def infer(self, input):
        return self.model(input).numpy()