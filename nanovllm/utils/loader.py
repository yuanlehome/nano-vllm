import os
from abc import ABC, abstractmethod
from glob import glob

import torch
from safetensors import safe_open
from torch import nn

from nanovllm.config import Config
from nanovllm.models import ModelRegistry


def get_model_from_loader(config: Config) -> nn.Module:
    """load or download model"""
    model_loader = DefaultModelLoader(config)
    model = model_loader.load_model()
    return model


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def download_model(self, config: Config):
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_model(self) -> nn.Module:
        """Load a model with the given configurations."""
        raise NotImplementedError


class DefaultModelLoader(BaseModelLoader):
    """ModelLoader that can load registered models"""

    def __init__(self, config: Config):
        super().__init__(config)

    def download_model(self, config: Config):
        raise NotImplementedError

    def load_model(self) -> nn.Module:
        model_class = ModelRegistry.get_class(self.config.hf_config.architectures[0])
        model = model_class(self.config.hf_config)

        packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
        for file in glob(os.path.join(self.config.model, "*.safetensors")):
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    for k in packed_modules_mapping:
                        if k in weight_name:
                            v, shard_id = packed_modules_mapping[k]
                            param_name = weight_name.replace(k, v)
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, f.get_tensor(weight_name), shard_id)
                            break
                    else:
                        param = model.get_parameter(weight_name)
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, f.get_tensor(weight_name))

        return model
