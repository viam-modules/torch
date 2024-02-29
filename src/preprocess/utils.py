from .vision import Preprocessor as VisionPreprocessor

def get_preprocessor(model_family):
    
    if model_family == 'vision':
        return VisionPreprocessor()