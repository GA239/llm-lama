# llm-knowledge-base

## Installation

```bash
make install
```

Note: If you are using Apple Silicon (M1) Mac, make sure you have installed a version of Python that supports arm64 architecture. For example:
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh

```
Otherwise, while installing it will build the llama.ccp x86 version which will be 10x slower on Apple Silicon (M1) Mac.

## Windows Installation

* (Optional) install and activate [pyenv-win](https://rkadezone.wordpress.com/2020/09/14/pyenv-win-virtualenv-windows/) 
* Install dependencies 
  * `pip install -r requirements.txt`
  * `pip install llama-cpp-python` if compiler and Cmake are installed
  * Already compiled whl for your system configuration `pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.6/llama_cpp_python-0.2.6-cp39-cp39-win_amd64.whl`
* `cd src`
* Download model
  * `python downloader.py`
  * Download model file from [here](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/tree/main), for example `llama-2-7b-chat.Q4_K_M.gguf`
* Run smoke test
  * python runner.py
* Run KB Retriever
  * `python small_doc_qa_draft.py` It takes some time for the first time to download models in cache.