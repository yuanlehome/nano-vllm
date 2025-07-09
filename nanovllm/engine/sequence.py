from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    counter = count()
    block_size = 256

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, key: int) -> int:
        return self.token_ids[key]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> bool:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self) -> list[int]:
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self) -> int:
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i) -> list[int]:
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self) -> list[int]:
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state: list[int]):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]

    def __repr__(self) -> str:
        return (
            f"  <Sequence id={self.seq_id}, status={self.status.name}, "
            f"num_tokens={self.num_tokens}, num_prompt_tokens={self.num_prompt_tokens}, "
            f"num_completion_tokens={self.num_completion_tokens}, num_cached_tokens={self.num_cached_tokens}, "
            f"num_blocks={self.num_blocks}, num_cached_blocks={self.num_cached_blocks}, "
            f"prompt_token_ids={self.prompt_token_ids}, completion_token_ids={self.completion_token_ids}, "
            f"last_block_num_tokens={self.last_block_num_tokens}, last_token={self.last_token}, max_tokens={self.max_tokens}, "
            f"ignore_eos={self.ignore_eos}, temperature={self.temperature:.2f}>"
        )
