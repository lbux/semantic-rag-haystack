import os

import torch
from haystack import Pipeline
from haystack.components.builders import AnswerBuilder, ChatPromptBuilder
from haystack.components.converters import (
    MarkdownToDocument,
    PyPDFToDocument,
    TextFileToDocument,
)
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import ComponentDevice
from haystack_integrations.components.generators.llama_cpp import LlamaCppChatGenerator

folder_path = "documents/source_documents"


def pretty_print_results(prediction):
    for doc in prediction["documents"]:
        print(doc.meta["file_path"], "\t", doc.score)
        print(doc.content)
        print("\n")


source_documents_folder = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

document_store = InMemoryDocumentStore()
document_splitter = DocumentSplitter(
    split_by="word", split_length=512, split_overlap=32
)
document_embedder = SentenceTransformersDocumentEmbedder(
    model="BAAI/bge-small-en-v1.5", device=ComponentDevice.from_str("cuda:0")
)
document_writer = DocumentWriter(document_store)
file_router = FileTypeRouter(
    mime_types=["application/pdf", "text/markdown", "text/plain"]
)


index_pipeline = Pipeline()
index_pipeline.add_component("document_splitter", document_splitter)
index_pipeline.add_component("document_embedder", document_embedder)
index_pipeline.add_component("document_writer", document_writer)
index_pipeline.add_component("document_cleaner", DocumentCleaner())
index_pipeline.add_component("document_joiner", DocumentJoiner())
index_pipeline.add_component("file_router", file_router)
index_pipeline.add_component("pdf_converter", PyPDFToDocument())
index_pipeline.add_component("markdown_converter", MarkdownToDocument())
index_pipeline.add_component("text_file_converter", TextFileToDocument())

index_pipeline.connect("file_router.text/plain", "text_file_converter.sources")
index_pipeline.connect("file_router.text/markdown", "markdown_converter.sources")
index_pipeline.connect("file_router.application/pdf", "pdf_converter.sources")
index_pipeline.connect("text_file_converter", "document_joiner")
index_pipeline.connect("markdown_converter", "document_joiner")
index_pipeline.connect("pdf_converter", "document_joiner")
index_pipeline.connect("document_joiner", "document_cleaner")
index_pipeline.connect("document_cleaner", "document_splitter")
index_pipeline.connect("document_splitter", "document_embedder")
index_pipeline.connect("document_embedder", "document_writer")

index_pipeline.draw("visual_design/index_pipeline.png")
index_pipeline.run({"file_router": {"sources": source_documents_folder}})
print(document_store.count_documents())

text_embedder = SentenceTransformersTextEmbedder(
    model="BAAI/bge-small-en-v1.5", device=ComponentDevice.from_str("cuda:0")
)
embedding_retriever = InMemoryEmbeddingRetriever(document_store)
bm25_retriever = InMemoryBM25Retriever(document_store)
ranker = TransformersSimilarityRanker(model="BAAI/bge-reranker-base", top_k=3)

llm = LlamaCppChatGenerator(
    model="models/Meta-Llama-3-8B-Instruct-Q6_K.gguf",
    model_kwargs={"n_gpu_layers": -1, "n_predict": -1},
    generation_kwargs={"max_tokens": 500},
)

llm.warm_up()

system_message = ChatMessage.from_system(
    """
    Read the context provided and answer the question if possible. If you can not form an answer from the context, reply with "Nah".
    Context:
    {% for doc in documents %}
    {{ doc.content }}
    {% endfor %};
    """
)
user_message = ChatMessage.from_user("query: {{query}}")
assistent_message = ChatMessage.from_assistant("Answer: ")

chat_template = [system_message, user_message, assistent_message]

hybrid_retrieval = Pipeline()
hybrid_retrieval.add_component("text_embedder", text_embedder)
hybrid_retrieval.add_component("embedding_retriever", embedding_retriever)
hybrid_retrieval.add_component("bm25_retriever", bm25_retriever)
hybrid_retrieval.add_component("document_joiner", DocumentJoiner())
hybrid_retrieval.add_component("ranker", ranker)
hybrid_retrieval.add_component(
    "prompt_builder", ChatPromptBuilder(template=chat_template)
)
hybrid_retrieval.add_component("llm", llm)
hybrid_retrieval.add_component("answer_builder", AnswerBuilder())

hybrid_retrieval.connect("text_embedder", "embedding_retriever")
hybrid_retrieval.connect("bm25_retriever", "document_joiner")
hybrid_retrieval.connect("embedding_retriever", "document_joiner")
hybrid_retrieval.connect("document_joiner", "ranker")
hybrid_retrieval.connect("ranker", "prompt_builder")
hybrid_retrieval.connect("prompt_builder", "llm")
hybrid_retrieval.connect("llm.replies", "answer_builder.replies")
# hybrid_retrieval.connect("llm.meta", "answer_builder.meta")
hybrid_retrieval.connect("embedding_retriever", "answer_builder.documents")

query = "What's up doc?"

hybrid_retrieval.draw("visual_design/hybrid_retrieval.png")
result = hybrid_retrieval.run(
    {
        "text_embedder": {"text": query},
        "bm25_retriever": {"query": query},
        "prompt_builder": {"query": query},
        "answer_builder": {"query": query},
        "ranker": {"query": query},
    }
)

print(result)
