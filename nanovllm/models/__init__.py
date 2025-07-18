from nanovllm.models.qwen3 import Qwen3ForCausalLM


class ModelRegistry:
    """
    Used to register and retrieve model classes.
    """

    _registry = {}

    @classmethod
    def register(cls, architecture: str, model_class):
        """register model class"""
        cls._registry[architecture] = model_class

    @classmethod
    def get_class(cls, architecture: str):
        """get model class"""
        if architecture not in cls._registry:
            raise ValueError(f"Model '{architecture}' is not registered!")
        return cls._registry[architecture]


ModelRegistry.register("Qwen3ForCausalLM", Qwen3ForCausalLM)
