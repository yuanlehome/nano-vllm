import torch
from flashinfer.sampling import top_k_top_p_sampling_from_logits
from torch import nn

from nanovllm.sample.metadata import SamplingMetadata

_SAMPLING_EPS = 1e-5

class Sampler(nn.Module):


    def __init__(self):
        super().__init__()

    def apply_temperature(
        self,
        logits: torch.Tensor,
        temp: torch.Tensor,
    ) -> torch.Tensor:
        # Use in-place division to avoid creating a new tensor.
        return logits.div_(temp.unsqueeze(dim=1))

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)

    def topk_topp_sampler(
        self,
        logits: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from the logits using FlashInfer.
        """
        # Both top-k and top-p.
        next_token_ids = top_k_top_p_sampling_from_logits(
            logits, top_k, top_p, deterministic=True)

        return next_token_ids.view(-1)

    def forward(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = logits.to(torch.float)
        greedy_tokens = self.greedy_sample(logits)
        if sampling_metadata.all_greedy:
            return greedy_tokens

        # Apply temperature.
        logits = self.apply_temperature(logits, sampling_metadata.temperature)
        # Apply top_k and top_p.
        sample_tokens = self.topk_topp_sampler(
            logits,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_tokens,
            sample_tokens,
            out=greedy_tokens, # Reuse tensor
        )
        return sampled
