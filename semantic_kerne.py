from PyPDF2 import PdfReader
from semantic_kernel.text import text_chunker as tc
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextCompletion, AzureTextEmbedding

def get_pdf_text(pdf_path):
    text_list = []
    try:       
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text_list.append(page.extract_text())
    except Exception as e:
        print(f"an error occurred while reading the PDF's: {str(e)}")
    
    return text_list

def get_chunk_text(pdf_text):
    chunk = tc.split_plaintext_paragraph(pdf_text, max_tokens=100)
    return chunk

def initialize_kernel():
    kernel = sk.Kernel()
    OPENAI_ENDPOINT = ""
    OPENAI_DEPLOYMENT_NAME = ""
    OPENAI_API_KEY = ""
    OPENAI_EMBEDDING_DEPLOYMENT_NAME = "text-embedding-ada-002"

    kernel.add_text_completion_service(
        service_id="azure_gpt35_text_completion",
        service=AzureTextCompletion(
            OPENAI_DEPLOYMENT_NAME, OPENAI_ENDPOINT, OPENAI_API_KEY, OPENAI_API_VERSION 
        ),
    )

    gpt35_chat_service = AzureChatCompletion(
        deployment_name=OPENAI_DEPLOYMENT_NAME,
        endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION,
    )

    kernel.add_chat_service("azure_gpt35_chat_completion", gpt35_chat_service)

    kernel.add_text_embedding_generation_service(
        "azure_openai_embedding",
        AzureTextEmbedding(
            deployment_name=OPENAI_EMBEDDING_DEPLOYMENT_NAME,
            endpoint=OPENAI_ENDPOINT,
            api_key=OPENAI_API_KEY,
        ),
    )

    return kernel

def register_memory_store(kernel):
    memory_store = sk.memory.VolatileMemoryStore()
    kernel.register_memory_store(memory_store=memory_store)

def search_questions(kernel):
    while True:
        user_input = input("Enter your question (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        
        results = kernel.memory.search(
            "Finance", user_input, limit=2
        )
        print(f"Results for '{user_input}':")
        for result in results:
            print(f"Text: {result.text} \nRelevance: {result.relevance}\n")

def main():
    pdf_path = input("Enter the path to the PDF file: ")
    pdf_text = get_pdf_text(pdf_path)
    pdf_chunks = get_chunk_text(pdf_text)
    kernel = initialize_kernel()
    register_memory_store(kernel)
    search_questions(kernel)

if __name__ == "__main__":
    main()
