import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Responda as perguntas com base somente no contexto a seguir:

{context}

---

Responda as perguntas com base no contexto acima: {question}
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Load the model outside the loop.
    model = Ollama(model="LukasM")

    # Batch process the documents.
    sources = [(doc.metadata.get("id", None), doc.page_content) for doc, _score in results]
    formatted_sources = "\n\n".join([f"ID: {id}\n\nContent:\n{content}" for id, content in sources])
    
    # Batch process the prompts.
    response_text = model.invoke(prompt)

    formatted_response = f"\n\nResponse:\n{response_text}\n"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()
