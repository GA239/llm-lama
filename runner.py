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


if __name__ == "__main__":
    m_name = "Llama-2-13B-GGML"
    model_file = "llama-2-13b.ggmlv3.q4_0.bin"

    # m_name = "Llama-2-7B-GGML"
    # model_file = "llama-2-7b.ggmlv3.q2_K.bin"

    # Super slow
    # model_path = get_local_model_path(m_name)
    # llm = AutoModelForCausalLM.from_pretrained(
    #     model_path_or_repo_id=model_path,
    #     model_type='llama',
    #     model_file=model_file,
    #     # TODO: try with config
    # )
    # print(llm('Q: Name the planets in the solar system? A:'))

    # # works
    model_path = os.path.join(get_local_model_path(m_name), model_file)
    llm = Llama(model_path=model_path, n_threads=7, use_mlock=True)
    output = llm("Q: Name the planets in the solar system? A: ", stop=["Q:", "\n"], echo=True)
    print(output)
    # output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)
    # print(output)


