from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.chroma import ChromaDocumentStore


document_store = ChromaDocumentStore(persist_path="chroma_test")

indexer_pipeline = Pipeline()

indexer_pipeline.add_component("converter", PyPDFToDocument())
indexer_pipeline.add_component(
    "document_embedder",
    SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    ),
)
indexer_pipeline.add_component(
    "document_writer", DocumentWriter(document_store=document_store)
)
indexer_pipeline.connect("converter", "document_embedder")
indexer_pipeline.connect("document_embedder", "document_writer")

indexer_pipeline.run({"converter": {"sources": ["source_documents/nothing.pdf"]}})
