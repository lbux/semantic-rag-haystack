from haystack import Pipeline
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from utils import serialize_generated_answer

chat_template = """
You are a helpful teaching assistant for a college course. Your objective
is to provide answers from the syllabus (context) to student questions (query).
If you are unable to answer a question based on the information in the syallabus,
you should respond with "I'm not sure".
Context:
{% for doc in documents %}
  {{ doc.content }}
{% endfor %}
query: {{query}}
Answer:
"""

generator = LlamaCppGenerator(
    model="models/mistral-7b-instruct-v0.2.Q6_K.gguf",
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
rag_pipeline.add_component("answer_builder", AnswerBuilder())


rag_pipeline.connect("text_embedder", "embedder_retriever")
rag_pipeline.connect("embedder_retriever", "prompt_builder")
rag_pipeline.connect("prompt_builder", "llm")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
rag_pipeline.connect("llm.meta", "answer_builder.meta")
rag_pipeline.connect("embedder_retriever", "answer_builder.documents")


rag_pipeline.draw("rag_pipeline.png")

prompts = [
    'Is "The user has to select the options A and B or C" an example of an ambiguous requirement?',
]
results = []
for prompt in prompts:
    result = rag_pipeline.run(
        {
            "text_embedder": {"text": prompt},
            "prompt_builder": {"query": prompt},
            "answer_builder": {"query": prompt},
        }
    )
    results.append(result)

serialize_generated_answer(results)
