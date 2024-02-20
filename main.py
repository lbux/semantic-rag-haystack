from getpass import getpass
import os

from haystack import Pipeline
from haystack.components.builders import PromptBuilder, AnswerBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from haystack_integrations.components.evaluators.deepeval import (
    DeepEvalEvaluator,
    DeepEvalMetric,
)


def evaluate_context_relevance(questions, evaluation_pipeline, rag_pipeline):
    contexts = []
    responses = []
    for question in questions:
        response = rag_pipeline.run(
            {
                "text_embedder": {"text": question},
                "prompt_builder": {"query": question},
                "answer_builder": {"query": question},
            }
        )
        contexts.append(
            [d.content for d in response["answer_builder"]["answers"][0].documents]
        )
        responses.append(response["answer_builder"]["answers"][0].data)

    evaluation_results = evaluation_pipeline.run(
        {
            "evaluator": {
                "questions": questions,
                "contexts": contexts,
                "responses": responses,
            }
        }
    )
    return evaluation_results


os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")

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
rag_pipeline.add_component("answer_builder", AnswerBuilder())


rag_pipeline.connect("text_embedder", "embedder_retriever")
rag_pipeline.connect("embedder_retriever", "prompt_builder")
rag_pipeline.connect("prompt_builder", "llm")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
rag_pipeline.connect("llm.meta", "answer_builder.meta")
rag_pipeline.connect("embedder_retriever", "answer_builder.documents")


rag_pipeline.draw("rag_pipeline.png")

# prompt = f"What is an LLM?"
# result = rag_pipeline.run(
#     {
#         "text_embedder": {"text": prompt},
#         "prompt_builder": {"query": prompt},
#         "answer_builder": {"query": prompt},
#     }
# )
# generated_text = result["llm"]["replies"][0]
# print(generated_text)

evaluator = DeepEvalEvaluator(
    metric=DeepEvalMetric.FAITHFULNESS,
    metric_params={"model": "gpt-3.5-turbo"},
)

evaluator_pipeline = Pipeline()
evaluator_pipeline.add_component("evaluator", evaluator)

questions = ["What is the grading of this class?", "What textbooks do I need?"]
print(evaluate_context_relevance(questions, evaluator_pipeline, rag_pipeline))
