import asyncio
from PyPDF2 import PdfReader
import semantic_kernel as sk
from semantic_kernel.text import text_chunker as tc
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextCompletion, AzureTextEmbedding

def get_pdf_text(pdf_path):
    print("Reading and extracting the texts from the pdf")
    text_list = []
    try:       
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text_list.append(page.extract_text())
        print("Done Reading and extracting the texts from the pdf")
    except Exception as e:
        print(f"an error occurred while reading the PDF's: {str(e)}")
    
    return text_list

def get_chunk_text(pdf_text):
    print("Initiating text chunking process")
    try:
        chunk = tc.split_plaintext_paragraph(pdf_text, max_tokens=100)
        print("Completed the chunking process")
        return chunk
    except Exception as e:
        print(f"An error occurred while chunking text: {str(e)}")
        return []

def initialize_kernel():
    kernel = sk.Kernel()
    print("Initializing Azure OpenAI and Embeddings")
    OPENAI_ENDPOINT = "https://openai-ppcazure017.openai.azure.com/"
    OPENAI_DEPLOYMENT_NAME = "gpt-35-turbo"
    OPENAI_EMBEDDING_DEPLOYMENT_NAME = "text-embedding-ada-002"
    OPENAI_API_KEY = "c3417acc5c654125b0b74cffeea8b491"
    OPENAI_API_TYPE = 'azure'
    OPENAI_API_VERSION = "2023-03-15-preview"

    try:
        print("Initializing the kernel")
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
        print("Completed initializing the kernel")
    except Exception as e:
        print(f"An error occurred while initializing kernel: {str(e)}")
    return kernel

def register_memory_store(kernel):
    try:
        print("Initializing memory store")
        memory_store = sk.memory.VolatileMemoryStore()
        kernel.register_memory_store(memory_store=memory_store)
        print("completed memory initialization process")
    except Exception as e:
        print(f"An error occurred while registering memory store: {str(e)}")

async def search_questions(kernel):
    try:
        while True:
            user_input = input("Enter your question (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            
            result = await kernel.memory.search_async(
                "Finance", user_input, limit=1
            )
            print(result[0].text)

    except KeyboardInterrupt:
        print("Process interrupted by user.")
    except Exception as e:
        print(f"An error occurred while searching questions: {str(e)}")

async def process_chunks_async(kernel, pdf_chunks):
    try:
        print("Initializing the embeddings creation process")
        for i, pdf_chunks in enumerate(pdf_chunks):
            await kernel.memory.save_information_async(
            collection="Finance", id="pdf_chunks" + str(i), text=pdf_chunks
        )
        print("Embeddings created successfully!")
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
        await search_questions(kernel)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main_async())
