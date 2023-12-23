from utils import download_model

if __name__ == "__main__":
    download_model(
        "TheBloke/Llama-2-7b-Chat-GGUF", "../models",
        ignore_patterns=[
            "llama-2-7b-chat.Q2*",
            "llama-2-7b-chat.Q3*",
            "llama-2-7b-chat.Q5*",
            "llama-2-7b-chat.Q6*",
            "llama-2-7b-chat.Q8*",
            "llama-2-7b-chat.Q4_0*",
            "llama-2-7b-chat.Q4_0*",
            "llama-2-7b-chat.Q4_K_S*",
            "llama-2-7b-chat.Q4_K_M*",
        ]
    )
    # download_model(
    #     "TheBloke/Llama-2-13B-chat-GGUF",
    #     "../models",
    #     ignore_patterns=[
    #         "llama-2-13b-chat.Q2*",
    #         "llama-2-13b-chat.Q3*",
    #         "llama-2-13b-chat.Q5*",
    #         "llama-2-13b-chat.Q6*",
    #         "llama-2-13b-chat.Q8*",
    #     ],
    # )
    # download_model(
    #     "TheBloke/Llama-2-70B-GGUF",
    #     "../models",
    #     ignore_patterns=[
    #         "llama-2-70b.Q2*",
    #         "llama-2-70b.Q3*",
    #         "llama-2-70b.Q5*",
    #         "llama-2-70b.Q6*",
    #         "llama-2-70b.Q8*",
    #         "llama-2-70b.Q4_K_S.gguf",
    #         "llama-2-70b.Q4_0.gguf",
    #     ],
    # )
