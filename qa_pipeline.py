# qa_pipeline.py
import logging

from pathlib import Path
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.dataclasses import ChatMessage
from haystack.utils import ComponentDevice
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
import os
from dotenv import load_dotenv

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
load_dotenv()

LLM_MODEL = os.getenv('LLM_MODEL')


def get_qa_pipeline():
    chroma_dir = "chroma_dir"
    # Load ChromaDocumentStore (saved in "chroma_dir" directory)
    document_store = ChromaDocumentStore(persist_path=chroma_dir, distance_function="cosine")

    # Prompt template for LLM: context is received from similar documents searched by user's question
    template = [
        ChatMessage.from_user("""
        ТИ – чат-бот/помічник факультету інформатики і обчислювальної техніки (ФІОТ)
        Національного університету Уркаїни "Київський Полетехнічний існтитут" (КПІ) ім. Ігоря Сікорського. Твоє завдання – надавати КОРЕКТНІ відповіді на
        питання КОРИСТУВАЧА, використовуючи наданий КОНТЕКСТ.
    
        ПРАВИЛА ТВОЄЇ РОБОТИ:
        1. Відповідай ТІЛЬКИ українською мовою, повними реченнями.
        2. Не використовуй форматування markdown (заборонено *...*, ```...``` тощо).
        3. Використовуй ТІЛЬКИ інформацію з наданого КОНТЕКСТУ. Якщо немає потрібної інформації –
        попроси користувача сформулювати питання детальніше або по іншому.
        4. Не відповідай на питання, які не стосуються ФІОТ або КПІ.
        5. Не вигадуй додаткові дані; посилайся рівно на те, що є у КОНТЕКСТІ (зокрема роки,
        посилання, назви кафедр).
    
        Нижче наведено КОНТЕКСТ (витяги з документів та офіційних сайтів).
        Спочатку уважно його проаналізуй, а потім сформулюй зрозумілу, точну відповідь,
        дотримуючись правил вище.
    
        ===========================================================================================================
        КОНТЕКСТ:
        {% for document in documents %}
        {{ document.content }}
        {% endfor %}

        ===========================================================================================================
        ПИТАННЯ: {{ question }}
    
        ТВОЯ ВІДПОВІДЬ:
    """)]

    # Build QA Pipeline
    qa_pipeline = Pipeline()

    # you can use any sentence-transformer model from https://huggingface.co/sentence-transformers
    # for example sentence-transformers/all-roberta-large-v1
    embedding_model = "lang-uk/ukr-paraphrase-multilingual-mpnet-base"  # ukrainian fine-tuned model

    # use 'mps' if you have MacBook on M1...4, or 'cuda' for NVIDIA GPU
    # remove 'device' if you have unsupported gpu
    device = ComponentDevice.from_str("cuda")

    qa_pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model=embedding_model, device=device))
    qa_pipeline.add_component("retriever", ChromaEmbeddingRetriever(document_store=document_store, top_k=10))
    qa_pipeline.add_component("chat_prompt_builder", ChatPromptBuilder(template=template))

    # uncomment next component if you want to use OpenAI chat generator
    # qa_pipeline.add_component("llm", OpenAIChatGenerator())

    # comment next component if you want to use OpenAI chat generator
    qa_pipeline.add_component("llm", OllamaChatGenerator(
        model=LLM_MODEL, url="http://localhost:11434", generation_kwargs={"temperature": 0.7, 'top_k': 30, 'top_p': 0.8}
    ))

    # Connect components
    qa_pipeline.connect("embedder.embedding", "retriever.query_embedding")
    qa_pipeline.connect("retriever", "chat_prompt_builder.documents")
    qa_pipeline.connect("chat_prompt_builder.prompt", "llm.messages")

    # uncomment to visualize QA pipeline
    # qa_pipeline.draw(Path("qa_pipeline.png"))

    return qa_pipeline
