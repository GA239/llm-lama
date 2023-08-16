import os
import time
from ctransformers import AutoModelForCausalLM

from langchain import ConversationChain, PromptTemplate

from langchain import PromptTemplate
from llama_cpp import Llama

TEMPLATE = """
The following is a friendly conversation between a human and an AI. 
The AI provides very short and precise answers from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
You: {input}
AI:"""


def get_local_model_path(model_name) -> str:
    path = "./models/TheBloke/"
    model_path = os.path.join(path, model_name)
    return model_path


def get_local_lama_model(model_name, **kwargs):
    model_path = get_local_model_path(model_name)
    return CTransformers(
        model=model_path,
        model_type='llama',
        model_file='llama-2-13b.ggmlv3.q6_K.bin',
        config=kwargs
    )


def with_ctransformers():
    # Super slow
    model_path = get_local_model_path(m_name)
    llm = AutoModelForCausalLM.from_pretrained(
        model_path_or_repo_id=model_path,
        model_type='llama',
        model_file=model_file,
        # TODO: try with config
    )
    print(llm('Q: Name the planets in the solar system? A:'))


if __name__ == "__main__":
    m_name = "Llama-2-7B-GGML"
    model_file = "llama-2-7b.ggmlv3.q4_K_M.bin"
    # output >>>
# llama_print_timings:        load time =  1570.52 ms
# llama_print_timings:      sample time =    23.36 ms /    32 runs   (    0.73 ms per token,  1370.04 tokens per second)
# llama_print_timings: prompt eval time =  1570.47 ms /    15 tokens (  104.70 ms per token,     9.55 tokens per second)
# llama_print_timings:        eval time =  6518.96 ms /    31 runs   (  210.29 ms per token,     4.76 tokens per second)
# llama_print_timings:       total time =  8153.39 ms -> 8 secs

    # m_name = "Llama-2-13B-GGML"
    # model_file = "llama-2-13b.ggmlv3.q4_0.bin"
    # output >>>
# llama_print_timings:        load time =  3117.50 ms
# llama_print_timings:      sample time =    23.70 ms /    32 runs   (    0.74 ms per token,  1350.15 tokens per second)
# llama_print_timings: prompt eval time =  3117.39 ms /    15 tokens (  207.83 ms per token,     4.81 tokens per second)
# llama_print_timings:        eval time = 11238.58 ms /    31 runs   (  362.53 ms per token,     2.76 tokens per second)
# llama_print_timings:       total time = 14424.64 ms -> 14 secs

    # m_name = "Llama-2-70B-GGML"
    # model_file = "llama-2-70b.ggmlv3.q4_0.bin"
    # output >>>
# llama_print_timings:        load time = 55380.01 ms
# llama_print_timings:      sample time =    23.84 ms /    32 runs   (    0.75 ms per token,  1342.00 tokens per second)
# llama_print_timings: prompt eval time = 55379.97 ms /    15 tokens ( 3692.00 ms per token,     0.27 tokens per second)
# llama_print_timings:        eval time = 1474075.16 ms /    31 runs   (47550.81 ms per token,   0.02 tokens per second)
# llama_print_timings:       total time = 1529580.91 ms  -> 25 mins

    # # works
    model_path = os.path.join(get_local_model_path(m_name), model_file)
    model_depend_args = {
        "Llama-2-7B-GGML": {"n_gqa": 8}
    }
    llm = Llama(
        model_path=model_path, n_threads=7, n_gpu_layers=1,
        **model_depend_args.get(m_name, {"use_mlock": True}),
    )
    # output = llm("Q: Name the planets in the solar system? A: ", stop=["Q:", "\n"], echo=True)
    # print(output)
    output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)
    print(output)
