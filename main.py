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


chat_template = """
Answer the query based on the provided context.
If the context does not contain the answer, say 'Answer not found'.
Don't say anything else.
Context:
{% for doc in documents %}
  {{ doc.content }}
{% endfor %}
query: {{query}}
Answer:
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

document_store = ChromaDocumentStore(persist_path="chroma_test")


rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component(
    "retriever", ChromaQueryRetriever(document_store=document_store)
)
rag_pipeline.add_component("prompt_builder", PromptBuilder(template=chat_template))
rag_pipeline.add_component("llm", generator)


rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

rag_pipeline.draw("rag_pipeline.png")

prompt = f"What is an LLM?"
result = rag_pipeline.run(
    {
        "retriever": {"query": prompt},
        "text_embedder": {"text": prompt},
        "prompt_builder": {"query": prompt},
    }
)
generated_text = result["llm"]["replies"][0]
print(generated_text)
