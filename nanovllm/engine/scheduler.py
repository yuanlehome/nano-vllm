from collections import deque

from nanovllm import envs
from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(
                seq
            ) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            if envs.NANOVLLM_ENABLE_DEBUG:
                print(f"\n[Prefill] before schedule for {seq=}, {self}")
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
            if envs.NANOVLLM_ENABLE_DEBUG:
                print(f"\n[Prefill] after schedule for {seq=}, {self}")
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            if envs.NANOVLLM_ENABLE_DEBUG:
                print(f"\n[Decode] before schedule for {seq=}, {self}")
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
            if envs.NANOVLLM_ENABLE_DEBUG:
                print(f"\n[Decode] after schedule for {seq=}, {self}")
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        assert len(seqs) == len(token_ids)
        for seq, token_id in zip(seqs, token_ids):
            if envs.NANOVLLM_ENABLE_DEBUG:
                print(f"\n[Postprocess] before for {seq=}, {self}")
            seq.append_token(token_id)
            if (
                not seq.ignore_eos and token_id == self.eos
            ) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
            if envs.NANOVLLM_ENABLE_DEBUG:
                print(f"\n[Postprocess] after for {seq=}, {self}")

    def __repr__(self):
        return (
            f"\nScheduler(\n"
            f"  max_num_seqs={self.max_num_seqs},\n"
            f"  max_num_batched_tokens={self.max_num_batched_tokens},\n"
            f"  eos={self.eos},\n"
            f"  block_manager:\n{repr(self.block_manager)},\n"
            f"  waiting=[\n    "
            + ",\n    ".join(repr(seq) for seq in self.waiting)
            + "\n  ],\n"
            "  running=[\n    "
            + ",\n    ".join(repr(seq) for seq in self.running)
            + "\n  ]\n)"
        )
