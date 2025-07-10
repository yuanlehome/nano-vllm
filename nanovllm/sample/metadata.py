from dataclasses import dataclass

import torch


@dataclass
class SamplingMetadata:

    temperature: torch.Tensor
    all_greedy: bool

    top_p: torch.Tensor
    top_k: torch.Tensor
