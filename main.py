import os

from haystack import Document, Pipeline
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.converters import PyPDFToDocument
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack_integrations.components.retrievers.chroma import ChromaQueryRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore


document_store = ChromaDocumentStore()

chat_template = """Given the following context, answer the question.

Context: 
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ question }}?
"""

generator = LlamaCppGenerator(
    model_path="models/mistral-7b-instruct-v0.2.Q6_K.gguf",
    n_ctx=32768,
    model_kwargs={"n_gpu_layers": -1},
    generation_kwargs={"max_tokens": 128, "temperature": 0.7},
)

text_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

generator.warm_up()
text_embedder.warm_up()


index_pipeline = Pipeline()

index_pipeline.add_component("converter", PyPDFToDocument())
index_pipeline.add_component(
    "document_embedder",
    SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    ),
)
index_pipeline.add_component(
    "document_writer", DocumentWriter(document_store=document_store)
)
index_pipeline.connect("converter", "document_embedder")
index_pipeline.connect("document_embedder", "document_writer")


rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component(
    "retriever", ChromaQueryRetriever(document_store=document_store)
)
rag_pipeline.add_component("prompt_builder", PromptBuilder(template=chat_template))
rag_pipeline.add_component("llm", generator)


# rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

rag_pipeline.draw("rag_pipeline.png")


index_pipeline.run({"converter": {"sources": ["source_documents/nothing.pdf"]}})

prompt = f"What is your name?"
result = rag_pipeline.run(
    {
        "retriever": {"query": prompt},
        "text_embedder": {"text": prompt},
        "prompt_builder": {"question": prompt},
    }
)
generated_text = result["llm"]["replies"][0]
print(generated_text)
