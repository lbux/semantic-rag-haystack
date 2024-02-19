import os
from haystack import Pipeline
from haystack.components.converters import (
    PyPDFToDocument,
    MarkdownToDocument,
    TextFileToDocument,
)
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

folder_path = "source_documents"

source_documents_folder = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]


document_store = ChromaDocumentStore(persist_path="chromaDB")

indexer_pipeline = Pipeline()

indexer_pipeline.add_component("pdf_converter", PyPDFToDocument())
indexer_pipeline.add_component(
    "document_embedder",
    SentenceTransformersDocumentEmbedder(model="hkunlp/instructor-large"),
)
indexer_pipeline.add_component(
    "document_writer", DocumentWriter(document_store=document_store)
)

indexer_pipeline.add_component(
    "file_type_router",
    FileTypeRouter(mime_types=["application/pdf", "text/markdown", "text/plain"]),
)
indexer_pipeline.add_component(
    "document_splitter",
    DocumentSplitter(split_by="word", split_length=150, split_overlap=50),
)
indexer_pipeline.add_component("document_cleaner", DocumentCleaner())
indexer_pipeline.add_component("document_joiner", DocumentJoiner())
indexer_pipeline.add_component("markdown_converter", MarkdownToDocument())
indexer_pipeline.add_component("text_file_converter", TextFileToDocument())


indexer_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
indexer_pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
indexer_pipeline.connect("file_type_router.application/pdf", "pdf_converter.sources")
indexer_pipeline.connect("text_file_converter", "document_joiner")
indexer_pipeline.connect("markdown_converter", "document_joiner")
indexer_pipeline.connect("pdf_converter", "document_joiner")
indexer_pipeline.connect("document_joiner", "document_cleaner")
indexer_pipeline.connect("document_cleaner", "document_splitter")
indexer_pipeline.connect("document_splitter", "document_embedder")
indexer_pipeline.connect("document_embedder", "document_writer")

indexer_pipeline.draw("indexer_pipeline.png")
indexer_pipeline.run({"file_type_router": {"sources": source_documents_folder}})
