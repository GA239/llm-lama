from utils import download_model

if __name__ == "__main__":
    download_model(
        "TheBloke/Llama-2-13B-GGML", "models",
        ignore_patterns=[
            "llama-2-13b.ggmlv3.q2*",
            "llama-2-13b.ggmlv3.q3*",
            "llama-2-13b.ggmlv3.q8*",
        ]
    )

    # download_model("TheBloke/Llama-2-7B-GGML", "models")

    # download_model(
    #     "TheBloke/Llama-2-70B-GGML", "models",
    #     ignore_patterns=[
    #         "llama-2-70b.ggmlv3.q2*",
    #         "llama-2-70b.ggmlv3.q3*",
    #         "llama-2-70b.ggmlv3.q5*",
    #         "llama-2-70b.ggmlv3.q6*",
    #         "llama-2-70b.ggmlv3.q8*",
    #         "llama-2-70b.ggmlv3.q4_1*",
    #         "llama-2-70b.ggmlv3.q4_K*",
    #     ]
    # )
