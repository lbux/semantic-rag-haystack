import os

from haystack import Pipeline
from haystack.components.converters import (
    MarkdownToDocument,
    PyPDFToDocument,
    TextFileToDocument,
)
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

folder_path = "source_documents"

source_documents_folder = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]


document_store = ChromaDocumentStore(persist_path="chromaDB")

ingest_pipeline = Pipeline()

ingest_pipeline.add_component("pdf_converter", PyPDFToDocument())
ingest_pipeline.add_component(
    "document_embedder",
    SentenceTransformersDocumentEmbedder(model="hkunlp/instructor-large"),
)
ingest_pipeline.add_component(
    "document_writer", DocumentWriter(document_store=document_store)
)

ingest_pipeline.add_component(
    "file_type_router",
    FileTypeRouter(mime_types=["application/pdf", "text/markdown", "text/plain"]),
)
ingest_pipeline.add_component(
    "document_splitter",
    DocumentSplitter(split_by="sentence", split_length=2, split_overlap=0),
)
ingest_pipeline.add_component("document_cleaner", DocumentCleaner())
ingest_pipeline.add_component("document_joiner", DocumentJoiner())
ingest_pipeline.add_component("markdown_converter", MarkdownToDocument())
ingest_pipeline.add_component("text_file_converter", TextFileToDocument())


ingest_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
ingest_pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
ingest_pipeline.connect("file_type_router.application/pdf", "pdf_converter.sources")
ingest_pipeline.connect("text_file_converter", "document_joiner")
ingest_pipeline.connect("markdown_converter", "document_joiner")
ingest_pipeline.connect("pdf_converter", "document_joiner")
ingest_pipeline.connect("document_joiner", "document_cleaner")
ingest_pipeline.connect("document_cleaner", "document_splitter")
ingest_pipeline.connect("document_splitter", "document_embedder")
ingest_pipeline.connect("document_embedder", "document_writer")

ingest_pipeline.draw("ingest_pipeline.png")
ingest_pipeline.run({"file_type_router": {"sources": source_documents_folder}})
