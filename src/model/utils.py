VISION_TYPES = ["object_detector"]
TEXT_TYPES = []
MODEL_TYPES = VISION_TYPES + TEXT_TYPES


def get_types():
    return MODEL_TYPES


def get_family(model_type):
    if model_type in VISION_TYPES or model_type =='vision':
        return 'vision'
    else:
        raise ValueError('Not ready for this kind of model')