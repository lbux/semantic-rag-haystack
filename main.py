from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
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
    n_ctx=12000,
    model_kwargs={"n_gpu_layers": -1},
    generation_kwargs={"max_tokens": 128, "temperature": 0.7},
)

text_embedder = SentenceTransformersTextEmbedder(model="hkunlp/instructor-large")

generator.warm_up()
text_embedder.warm_up()

document_store = ChromaDocumentStore(persist_path="chromaDB")


rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component(
    "embedder_retriever", ChromaEmbeddingRetriever(document_store=document_store)
)
rag_pipeline.add_component("prompt_builder", PromptBuilder(template=chat_template))
rag_pipeline.add_component("llm", generator)


rag_pipeline.connect("text_embedder", "embedder_retriever")
rag_pipeline.connect("embedder_retriever", "prompt_builder")
rag_pipeline.connect("prompt_builder", "llm")

rag_pipeline.draw("rag_pipeline.png")

prompt = f"What is an LLM?"
result = rag_pipeline.run(
    {
        "text_embedder": {"text": prompt},
        "prompt_builder": {"query": prompt},
    }
)
generated_text = result["llm"]["replies"][0]
print(generated_text)
