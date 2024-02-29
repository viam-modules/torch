import torch
from typing import List, Tuple
from numpy.typing import NDArray
import numpy as np

class TorchModel:
    def __init__(self,
                 path_to_serialized_file:str, 
                 path_to_model_file:str=None) -> None:
        if path_to_model_file is None:
            self.model = torch.jit.load(path_to_serialized_file)
        else:
            raise NotImplementedError("not yet supporting uploading model from file and weights")
        self.model.eval()
    
    def infer(self, input):
        return self.model(input).numpy()