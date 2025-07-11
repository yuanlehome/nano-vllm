from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


def main():
    model = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B/"
    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = LLM(
        model=model,
        max_model_len=8192,
        enforce_eager=True,
        tensor_parallel_size=2,
    )

    sampling_params = SamplingParams(temperature=0.0, top_p=0.95, top_k=20, max_tokens=3)
    prompts = [
        "请详细说明 Python 中多线程（threading）与多进程（multiprocessing）之间的本质区别，包括它们在资源占用、执行方式、调度机制、数据共享、以及 GIL（全局解释器锁）对其影响方面的异同。此外，结合你实际项目中的经验，说明你在什么场景下更倾向于使用多进程，哪些情况下会选择多线程，并说明背后的权衡逻辑。请详细说明 Python 中多线程（threading）与多进程（multiprocessing）之间的本质区别，包括它们在资源占用、执行方式、调度机制、数据共享、以及 GIL（全局解释器锁）对其影响方面的异同。此外，结合你实际项目中的经验，说明你在什么场景下更倾向于使用多进程，哪些情况下会选择多线程，并说明背后的权衡逻辑。请详细说明 Python 中多线程（threading）与多进程（multiprocessing）之间的本质区别，包括它们在资源占用、执行方式、调度机制、数据共享、以及 GIL（全局解释器锁）对其影响方面的异同。此外，结合你实际项目中的经验，说明你在什么场景下更倾向于使用多进程，哪些情况下会选择多线程，并说明背后的权衡逻辑。",
        "假设你要为一个分布式机器学习训练平台或在线推理系统设计日志系统，要求支持异步记录、日志等级控制、跨服务聚合、按天/大小滚动、可搜索查询等功能。请你详细描述整体架构设计，日志采集、处理、存储方案的选型（如是否使用 ELK、FluentBit、Kafka 等），并讨论如何实现低延迟、高可靠性和安全性，尤其在高并发和异常频发的环境下的优化措施。",
        "请详细说明 Python 中多线程（threading）与多进程（multiprocessing）之间的本质区别，包括它们在资源占用、执行方式、调度机制、数据共享、以及 GIL（全局解释器锁）对其影响方面的异同。此外，结合你实际项目中的经验，说明你在什么场景下更倾向于使用多进程，哪些情况下会选择多线程，并说明背后的权衡逻辑。请详细说明 Python 中多线程（threading）与多进程（multiprocessing）之间的本质区别，包括它们在资源占用、执行方式、调度机制、数据共享、以及 GIL（全局解释器锁）对其影响方面的异同。此外，结合你实际项目中的经验，说明你在什么场景下更倾向于使用多进程，哪些情况下会选择多线程，并说明背后的权衡逻辑。请详细说明 Python 中多线程（threading）与多进程（multiprocessing）之间的本质区别，包括它们在资源占用、执行方式、调度机制、数据共享、以及 GIL（全局解释器锁）对其影响方面的异同。此外，结合你实际项目中的经验，说明你在什么场景下更倾向于使用多进程，哪些情况下会选择多线程，并说明背后的权衡逻辑。",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
