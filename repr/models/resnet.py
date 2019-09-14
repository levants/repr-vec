"""
Created on Jul 12, 2019

Implementation of ResNet models with adaptive average pooling (image resizing tolerant)
for fine-tuning, transfer learning, features extractor etc

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from fastai.vision import *
from torch.utils import model_zoo
from torchvision.models.resnet import (ResNet, Bottleneck, BasicBlock, model_urls)

__all__ = ['Flatten', 'ResNetCore', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'init_model']


class Flatten(nn.Module):
    """Flatten input tensor to vector"""

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class ResNetCore(ResNet):
    """ResNet extension core model"""

    def __init__(self, block, layers, channels=3, num_classes=1000):
        super(ResNetCore, self).__init__(block, layers, num_classes=num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.flatten = Flatten()

    def features(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extracts features from input tensor
        Args:
            input_tensor: input image tensor

        Returns:
            repr_vec: features tensor
        """
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        repr_vec = self.layer4(x)

        return repr_vec

    def glob_pool(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extracts features from average pooling layer of input tensor
        Args:
            input_tensor: input image tensor

        Returns:
            repr_vec: features tensor
        """
        x = self.features(input_tensor)
        repr_vec = self.avgpool(x)

        return repr_vec

    def vect(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extracts features from average pooling layer of input tensor
        Args:
            input_tensor: input image tensor

        Returns:
            repr_vec: features tensor
        """
        x = self.glob_pool(input_tensor)
        repr_vec = self.flatten(x)

        return repr_vec

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.vect(input_tensor)
        logits = self.fc(x)

        return logits


def _init_layers(layers: list) -> list:
    """
    Sets default values to layers
    Args:
        layers: layers for ResNet module

    Returns:
        default value if layers are not defined
    """
    return [2, 2, 2, 2] if layers is None else layers


def _init_model(core_type: nn.Module = ResNetCore, block: nn.Module = BasicBlock, layers: list = None,
                model_key: str = 'resnet18', pretrained: bool = False, **kwargs) -> nn.Module:
    """
    Initializes appropriated model
    Args:
        core_type: type for model core initialization
        block: block for layers initialization
        layers: model layers
        model_key: key for model URL dictionary
        pretrained: flag for trained weights
        **kwargs: additional arguments

    Returns:
        model: network model with pre-trained weights
    """
    model = core_type(block, _init_layers(layers), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[model_key]))

    return model


def _init_module(block: nn.Module = BasicBlock, layers: list = None, model_key: str = 'resnet18',
                 pretrained: bool = False, **kwargs) -> nn.Module:
    """
    Initializes appropriated model by name
    Args:
        block: block for layers initialization
        layers: model layers
        model_key: model architecture
        pretrained: flag for trained weights
        **kwargs: additional arguments

    Returns:
        network model with pre-trained weights
    """
    return _init_model(core_type=ResNetCore, block=block, layers=layers, model_key=model_key, pretrained=pretrained,
                       **kwargs)


def resnet18(pretrained: bool = False, **kwargs) -> nn.Module:
    """
    Constructs a ResNet-18 model.
    Args:
        pretrained: if True, returns a model pre-trained on ImageNet data
        **kwargs: additional named arguments

    Returns:
        network model with pre-trained weights
    """
    return _init_module(block=BasicBlock, layers=[2, 2, 2, 2], model_key=resnet18.__name__, pretrained=pretrained,
                        **kwargs)


def resnet34(pretrained: bool = False, **kwargs) -> nn.Module:
    """
    Constructs a ResNet-34 model.
    Args:
        pretrained: if True, returns a model pre-trained on ImageNet data
        **kwargs: additional named arguments

    Returns:
        network model with pre-trained weights
    """
    return _init_module(block=BasicBlock, layers=[3, 4, 6, 3], model_key=resnet34.__name__, pretrained=pretrained,
                        **kwargs)


def resnet50(pretrained: bool = False, **kwargs) -> nn.Module:
    """
    Constructs a ResNet-50 model.
    Args:
        pretrained: If True, returns a model pre-trained on ImageNet data
        **kwargs: additional named arguments

    Returns:
        network model with pre-trained weights
    """
    return _init_module(block=Bottleneck, layers=[3, 4, 6, 3], model_key=resnet50.__name__, pretrained=pretrained,
                        **kwargs)


def resnet101(pretrained: bool = False, **kwargs) -> nn.Module:
    """
    Constructs a ResNet-101 model.
    Args:
        pretrained: If True, returns a model pre-trained on ImageNet data
        **kwargs: additional named arguments

    Returns:
        network model with pre-trained weights
    """
    return _init_module(block=Bottleneck, layers=[3, 4, 23, 3], model_key=resnet101.__name__, pretrained=pretrained,
                        **kwargs)


def resnet152(pretrained: bool = False, **kwargs) -> nn.Module:
    """
    Constructs a ResNet-152 model.
    Args:
        pretrained: If True, returns a model pre-trained on ImageNet data
        **kwargs: additional named arguments

    Returns:
        network model with pre-trained weights
    """
    return _init_module(block=Bottleneck, layers=[3, 8, 36, 3], model_key=resnet152.__name__, pretrained=pretrained,
                        **kwargs)


def _name_2_func(arch_name: str) -> callable:
    """
    Extracts callable by name
    Args:
        arch_name: architecture name

    Returns:
        arch_func: architecture callable
    """
    if arch_name == resnet18.__name__:
        arch_func = resnet18
    elif arch_name == resnet34.__name__:
        arch_func = resnet34
    elif arch_name == resnet50.__name__:
        arch_func = resnet50
    elif arch_name == resnet101.__name__:
        arch_func = resnet101
    elif arch_name == resnet152.__name__:
        arch_func = resnet152
    else:
        arch_func = resnet34

    return arch_func


def init_model(arch: str, pretrained: bool = False, head: nn.Sequential = None, **kwargs) -> nn.Module:
    """
    Initializes ResNet model
    Args:
        arch: architecture callable
        pretrained: pre-trained model
        head: custom header
        **kwargs: additional arguments

    Returns:
        model: initialized model
    """
    model_func = _name_2_func(arch) if isinstance(arch, str) else _name_2_func(arch.__name__)
    if head is None:
        model = model_func(pretrained=pretrained, **kwargs)
    else:
        body = create_body(model_func, pretrained=pretrained)
        model = nn.Sequential(body, head)

    return model
