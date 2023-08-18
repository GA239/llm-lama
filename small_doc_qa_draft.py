import os

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, NLTKTextSplitter
from runner import get_cpp_lama, get_local_model_path
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain import PromptTemplate, LLMChain
from langchain.chains.question_answering import load_qa_chain

URL = "https://luminousmen.com/post/github-pull-request-templates"
m_name = "Llama-2-13B-GGML"
model_file = "llama-2-13b.ggmlv3.q4_0.bin"
DOC_NUM = 3


def get_data(url=None):
    data_url = url or URL
    loader = WebBaseLoader(data_url)
    data = loader.load()
    return data


def split_text(data):
    text_splitter = NLTKTextSplitter(chunk_size=1500, chunk_overlap=1400)
    all_splits = text_splitter.split_documents(data)
    for split in all_splits:
        print(split)
    return all_splits


def vectorize_text(splits):
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=GPT4AllEmbeddings())
    return vectorstore


def test_vector_store(vectorstore: Chroma):
    question = "What are the Steps to Create an Effective Template?"
    docs = vectorstore.similarity_search_with_score(question, k=DOC_NUM)
    print(len(docs))
    for doc, score in docs:
        print(score, " " * 10, doc)


def get_model():
    model_path = os.path.join(get_local_model_path(m_name), model_file)
    return get_cpp_lama(m_name, model_path, lang_chain=True)


def create_chain(model):
    """
    Run an LLMChain with either model by passing in the retrieved docs and a simple prompt.

    It formats the prompt template using the input key values provided and passes the formatted string
    to LLama-V2, or another specified LLM.

    We can use a QA chain to handle our question above.
    chain_type="stuff" means that all the docs will be added (stuffed) into a prompt.
    """
    # Prompt
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use one sentence maximum and keep the answer as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:"""

    qa_chain_prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=template,
    )
    # Chain
    chain = load_qa_chain(model, chain_type="stuff", prompt=qa_chain_prompt)
    # todo: debug
    # chain = load_qa_chain(model, chain_type="map_reduce", question_prompt=qa_chain_prompt)
    return chain


def run_chain(chain, vectorstore, question):
    docs = vectorstore.similarity_search(question, k=DOC_NUM)
    print("::DEBUG::")
    print(len(docs))
    for doc in docs:
        print(doc)
    print("::DEBUG::")

    # Run
    return chain(
        {
            "input_documents": docs,
            "question": question
        },
        return_only_outputs=True
    )


def main():
    dt = get_data()
    sp_dt = split_text(dt)
    vs = vectorize_text(sp_dt)
    # test_vector_store(vs)
    ch = create_chain(get_model())
    # repl = run_chain(ch, vs, "Name the Steps to Create an Effective Template.")
    repl = run_chain(ch, vs, "What is a Pull Request Template?")
    print(repl)


if __name__ == "__main__":
    main()

# 1. Identify your team's needs: Consider what information and tasks need to be included in your pull request template. This can include a description of changes, relevant links and documentation, and any pre-review checklists or tests.
# 2. Use standard formatting: Use standard sections and headings, as well as consistent formatting for text and code blocks. This makes it easier for reviewers to quickly understand the content of your pull request template.
# 3. Test your template: Try out your template on a few existing pull requests to ensure that all necessary information is included and to identify any areas for improvement.
# 4. Revise as needed: Based on feedback from team members or your own experience, revise your template as needed to make it more effective. This may involve adding or removing sections, updating formatting guidelines, or making other changes to improve the overall quality of your pull request templates.
# 5. Share with your team: Once you have finalized your template, share it with your team members and encourage them to use it for all future pull requests. By using a consistent format, reviewers can quickly understand and evaluate the changes in each pull request, reducing confusion and improving communication within your team.