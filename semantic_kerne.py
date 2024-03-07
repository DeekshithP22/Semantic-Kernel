import asyncio
from PyPDF2 import PdfReader
from PyPDF2.utils import PdfReadError
from semantic_kernel.text import text_chunker as tc
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextCompletion, AzureTextEmbedding

def get_pdf_text(pdf_path):
    text_list = []
    try:       
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text_list.append(page.extract_text())
    except FileNotFoundError:
        print(f"Error: PDF file '{pdf_path}' not found.")
    except PdfReadError:
        print(f"Error: Unable to read PDF file '{pdf_path}'.")
    except Exception as e:
        print(f"An error occurred while reading the PDF: {str(e)}")
    
    return text_list

def get_chunk_text(pdf_text):
    try:
        chunk = tc.split_plaintext_paragraph(pdf_text, max_tokens=100)
        return chunk
    except Exception as e:
        print(f"An error occurred while chunking text: {str(e)}")
        return []

def initialize_kernel():
    kernel = sk.Kernel()
    OPENAI_ENDPOINT = ""
    OPENAI_DEPLOYMENT_NAME = ""
    OPENAI_API_KEY = ""
    OPENAI_EMBEDDING_DEPLOYMENT_NAME = "text-embedding-ada-002"

    try:
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
    except Exception as e:
        print(f"An error occurred while initializing kernel: {str(e)}")
    
    return kernel

def register_memory_store(kernel):
    try:
        memory_store = sk.memory.VolatileMemoryStore()
        kernel.register_memory_store(memory_store=memory_store)
    except Exception as e:
        print(f"An error occurred while registering memory store: {str(e)}")

def search_questions(kernel):
    try:
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
    except KeyboardInterrupt:
        print("Process interrupted by user.")
    except Exception as e:
        print(f"An error occurred while searching questions: {str(e)}")

async def process_chunks_async(kernel, pdf_chunks):
    try:
        for i, chunk in enumerate(pdf_chunks):
            embeddings = await kernel.memory.save_information_async(
                collection="Finance", id="pdf_chunks" + str(i), text=chunk
            )
    except Exception as e:
        print(f"An error occurred while processing chunks asynchronously: {str(e)}")

async def main_async():
    try:
        pdf_path = input("Enter the path to the PDF file: ")
        pdf_text = get_pdf_text(pdf_path)
        if not pdf_text:
            return

        pdf_chunks = get_chunk_text(pdf_text)
        if not pdf_chunks:
            return
        
        kernel = initialize_kernel()
        if not kernel:
            return

        register_memory_store(kernel)

        await process_chunks_async(kernel, pdf_chunks)
        search_questions(kernel)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main_async())
