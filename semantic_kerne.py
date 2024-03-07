from PyPDF2 import PdfReader
from semantic_kernel.text import text_chunker as tc
import semantic_kernel as sk
kernel = sk.Kernel()
OPENAI_ENDPOINT = ""
OPENAI_DEPLOYMENT_NAME = ""
OPENAI_EMBEDDING_DEPLOYMENT_NAME = ""
OPENAI_API_KEY = ""
OPENAI_API_TYPE = ''
OPENAI_API_VERSION = ""
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion,AzureTextCompletion

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
def get_pdf_text(pdf_path):
    text_list =[]
    try:       
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text_list.append(page.extract_text())
    except Exception as e:
        print(f"an error occured while reading the PDF's:{str(e)}")
    
    return text_list
def get_chunk_text(pdf_text):
    chunk = tc.split_plaintext_paragraph(pdf_text, max_tokens=100)
    return chunk
OPENAI_EMBEDDING_DEPLOYMENT_NAME = "text-embedding-ada-002"
from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding

kernel.add_text_embedding_generation_service(
    "azure_openai_embedding",
    AzureTextEmbedding(
        deployment_name=OPENAI_EMBEDDING_DEPLOYMENT_NAME,
        endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY,
    ),
)
memory_store = sk.memory.VolatileMemoryStore()
kernel.register_memory_store(memory_store=memory_store)
pdf_path = r"C:\Users\deekshith.p\Desktop\PDF\Basics_of_finmkts.pdf"
pdf_text = get_pdf_text(pdf_path)
pdf_text
pdf_chunks = get_chunk_text(pdf_text)
pdf_chunks
len(pdf_chunks)
for i, pdf_chunks in enumerate(pdf_chunks):
    embeddings = await kernel.memory.save_information_async(
        collection="Finance", id="pdf_chunks" + str(i), text=pdf_chunks
    )
context_embeddings = embeddings
results = await kernel.memory.search_async(
    "Finance", "What are the long term investment options?", limit=2
)
for result in results:
    print(f"Text: {result.text} \nRelevance:{result.relevance}\n")
results = await kernel.memory.search_async(
    "Finance", "Why should we invest", limit=2
)
results = await kernel.memory.search_async(
    "Finance", "Why should we invest"
)
print(result.text)
for result in results:
    print(f"Text: {result.text} \nRelevance:{result.relevance}\n")
