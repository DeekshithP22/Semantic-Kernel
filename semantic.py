from PyPDF2 import PdfReader
from semantic_kernel.text import text_chunker as tc
import semantic_kernel as sk
kernel = sk.Kernel()
import os
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
def get_pdf_text(folder_path):
    text_list =[]
    try: 
        for filename in os.listdir(folder_path):
            if filename.endswith('.pdf'):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    for page in pdf_reader.pages:
                        text_list.append(page.extract_text())
    except Exception as e:
        print(f"an error occured while reading the PDF's:{str(e)}")
    
    return text_list
def get_chunk_text(pdf_text):
    chunk = tc.split_plaintext_paragraph(pdf_text, max_tokens=1000)
    return chunk
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
pdf_path = r"C:\Users\deekshith.p\Desktop\PDF"
pdf_text = get_pdf_text(pdf_path)
pdf_text
pdf_chunks = get_chunk_text(pdf_text)
pdf_chunks
len(pdf_chunks)
for i, pdf_chunks in enumerate(pdf_chunks):
     await kernel.memory.save_information_async(
        collection="PDF", id="pdf_chunks" + str(i), text=pdf_chunks
    )
question = "Give me 5 lines summary on AI and Elections"
context = await kernel.memory.search_async(
    "PDF", question, limit=1
)
for result in context:
    print(f"Text: {result.text} \nRelevance:{result.relevance}\n")
hf_config_dict = {
    "schema": 1,
    # The type of prompt
    "type": "completion",
    # A description of what the semantic function does
    "description": "RAG using gpt35 model, which answers the user query based on the context provided",
    # Specifies which model service(s) to use
    "default_services": ["azure_gpt35_chat_completion"],
    # The parameters that will be passed to the connector and model service
    "completion": {
        "temperature": 0.5,
        "top_p": 1,
        "max_tokens": 1000,
        "number_of_responses": 1,
    },
    # Defines the variables that are used inside of the prompt
    "input": {
        "parameters": [
            {
                "name": "context",
                "description": "The context from which the bot has to answer",
                "defaultValue": "none",

                "name": "question",
                "description": "question the bot has to answer",
                "defaultValue": "none",
            }
        ]
    },
}

from semantic_kernel import PromptTemplateConfig

prompt_template_config = PromptTemplateConfig.from_dict(hf_config_dict)
from semantic_kernel import PromptTemplate

prompt_template = sk.PromptTemplate(
    template="""Answer the user's question as thoroughly as possible based on the provided 
            context. If there are any spelling mistakes in the user question, please correct them before searching 
            for the answer. Strive to locate the most relevant context for the query. If the answer is not available 
            in the provided context, respond with "Answer is not available in the context" without providing an 
            incorrect answer. \n\n Context:\n {{$context}}?\n Question:\n{{question}}\n\n Please analyze the user's 
            question and structure your response accordingly""",
    prompt_config=prompt_template_config,
    template_engine=kernel.prompt_template_engine,
)
from semantic_kernel import SemanticFunctionConfig

function_config = SemanticFunctionConfig(prompt_template_config, prompt_template)
from semantic_kernel import PromptTemplateConfig, SemanticFunctionConfig, PromptTemplate


def create_semantic_function_config(prompt_template, prompt_config_dict, kernel):
    prompt_template_config = PromptTemplateConfig.from_dict(prompt_config_dict)
    prompt_template = sk.PromptTemplate(
        template=prompt_template,
        prompt_config=prompt_template_config,
        template_engine=kernel.prompt_template_engine,
    )
    return SemanticFunctionConfig(prompt_template_config, prompt_template)
Chat_complete = kernel.register_semantic_function(
    skill_name="GPT35ChatComplete",
    function_name="chat_complete",
    function_config=create_semantic_function_config(
        """Answer the user's question as thoroughly as possible based on the provided 
            context. If there are any spelling mistakes in the user question, please correct them before searching 
            for the answer. Strive to locate the most relevant context for the query. If the answer is not available 
            in the provided context, respond with "Answer is not available in the context" without providing an 
            incorrect answer. \n\n Context:\n {{$context}}?\n Question:\n{{question}}\n\n Please analyze the user's 
            question and structure your response accordingly""", hf_config_dict, kernel
    ),
)
response = await Chat_complete.invoke_async(
    input = {"Context": context[0], "Question":question}
    )
print(response)
