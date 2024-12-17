import torch
import torchvision
import timm

vision_model_opts = {
    'resnet18': torchvision.models.resnet18,
    'resnet34': torchvision.models.resnet34,
    'resnet50': torchvision.models.resnet50,
    'resnet101': torchvision.models.resnet101,
}

model_weights = {
    'resnet18': torchvision.models.ResNet18_Weights,
    'resnet34': torchvision.models.ResNet34_Weights,
    'resnet50': torchvision.models.ResNet50_Weights,
    'resnet101': torchvision.models.ResNet101_Weights,
}


def construct_model(model_type, params=None):
    if model_type in vision_model_opts:
        weights = None
        pretrained = params['pretrained']
        if pretrained:
            weights = model_weights[model_type]
        model = vision_model_opts[model_type](weights=weights)
    elif model_type == 'timm_model':
        model = timm.create_model(params['model_name'], pretrained=params['pretrained'])

    else:
        raise NotImplementedError

    return model


def modify_model_output_layer(model, num_classes):
    if 'torchvision.models.resnet' in str(type(model)):
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif 'timm.models' in str(type(model)):
        if 'resnet' in str(type(model)):
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif 'vision_transformer' in str(type(model)):
            if hasattr(model.head, 'in_features'):
                model.head = torch.nn.Linear(model.head.in_features, num_classes)
            else:
                model.head = torch.nn.Linear(model.embed_dim, num_classes)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return model


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model
