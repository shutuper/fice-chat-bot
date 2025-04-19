# data_pipeline.py
import os
import shutil
from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import (
    HTMLToDocument,
    MarkdownToDocument,
    PyPDFToDocument,
    TextFileToDocument,
    DOCXToDocument
)
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.utils import ComponentDevice
from haystack_integrations.document_stores.chroma import ChromaDocumentStore


def build_index():
    chroma_dir = "chroma_dir"
    # Clear chroma_dir if exists
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)

    # Initialize DocumentStore (Chroma) for documents & embeddings storing
    document_store = ChromaDocumentStore(persist_path=chroma_dir, distance_function="cosine")

    # add supported document types
    file_type_router = FileTypeRouter(
        mime_types=[
            "text/plain",
            "application/pdf",
            "text/markdown",
            "text/html",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ],
        additional_mimetypes={"application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx"}
    )

    # Document converters initialization
    text_file_converter = TextFileToDocument()
    markdown_converter = MarkdownToDocument()
    pdf_converter = PyPDFToDocument()
    docx_converter = DOCXToDocument()
    html_converter = HTMLToDocument()

    # Document processors initialization
    document_joiner = DocumentJoiner()
    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(split_by="word", language="ru")

    # you can use can be used any sentence-transformer model from https://huggingface.co/sentence-transformers
    # for example sentence-transformers/all-roberta-large-v1
    embedding_model = "lang-uk/ukr-paraphrase-multilingual-mpnet-base"  # ukrainian fine-tuned model

    # use 'mps' if you have MacBook on M1...4, or 'cuda' for NVIDIA GPU
    # remove 'device' if you have unsupported gpu
    device = ComponentDevice.from_str("cuda")

    document_embedder = SentenceTransformersDocumentEmbedder(model=embedding_model, device=device)
    document_writer = DocumentWriter(document_store)

    # Build data processing pipeline
    preprocessing_pipeline = Pipeline()
    preprocessing_pipeline.add_component(instance=file_type_router, name="file_type_router")
    preprocessing_pipeline.add_component(instance=text_file_converter, name="text_file_converter")
    preprocessing_pipeline.add_component(instance=markdown_converter, name="markdown_converter")
    preprocessing_pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
    preprocessing_pipeline.add_component(instance=docx_converter, name="docx_converter")
    preprocessing_pipeline.add_component(instance=html_converter, name="html_converter")
    preprocessing_pipeline.add_component(instance=document_joiner, name="document_joiner")
    preprocessing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
    preprocessing_pipeline.add_component(instance=document_splitter, name="document_splitter")
    preprocessing_pipeline.add_component(instance=document_embedder, name="document_embedder")
    preprocessing_pipeline.add_component(instance=document_writer, name="document_writer")

    # Connect components:
    # file_type_router -> appropriate file_converter
    preprocessing_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
    preprocessing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
    preprocessing_pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
    preprocessing_pipeline.connect(
        "file_type_router.application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "docx_converter.sources")
    preprocessing_pipeline.connect("file_type_router.text/html", "html_converter.sources")

    # file_converter -> document_joiner
    preprocessing_pipeline.connect("text_file_converter", "document_joiner")
    preprocessing_pipeline.connect("pypdf_converter", "document_joiner")
    preprocessing_pipeline.connect("markdown_converter", "document_joiner")
    preprocessing_pipeline.connect("docx_converter", "document_joiner")
    preprocessing_pipeline.connect("html_converter", "document_joiner")

    # processing components
    preprocessing_pipeline.connect("document_joiner", "document_cleaner")
    preprocessing_pipeline.connect("document_cleaner", "document_splitter")
    preprocessing_pipeline.connect("document_splitter", "document_embedder")
    preprocessing_pipeline.connect("document_embedder", "document_writer")

    # uncomment to visualize pipeline
    # preprocessing_pipeline.draw(Path("data_pipeline.png"))

    # Launch a pipeline for all files in doc_dir directory
    doc_dir = "data/fiot_files"
    preprocessing_pipeline.run({"file_type_router": {"sources": list(Path(doc_dir).glob("**/*"))}})
