from torchvision import transforms

_PARAMS = {
    'resnet18': {
        'resize': 256,
        'crop':   224,
        'mean':   [0.485, 0.456, 0.406], # te parametry dla ImageNet
        'std':    [0.229, 0.224, 0.225],
    },
    'resnet50': {
        'resize': 232,
        'crop':   224,
        'mean':   [0.485, 0.456, 0.406],
        'std':    [0.229, 0.224, 0.225],
    },
    'vgg': {
        'resize': 256,
        'crop':   224,
        'mean':   [0.485, 0.456, 0.406],
        'std':    [0.229, 0.224, 0.225],
    },
}

def get_transforms(model_name, mean=None, std=None):
    if model_name not in _PARAMS:
        raise ValueError(f"Nieznany model: '{model_name}'. Dostępne: {list(_PARAMS.keys())}")
    p = _PARAMS[model_name]
    return transforms.Compose([
        transforms.Resize(p['resize']),
        transforms.CenterCrop(p['crop']),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean if mean is not None else p['mean'],
            std=std  if std  is not None else p['std'],
        ),
    ])
