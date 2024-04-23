from haystack import Pipeline
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.rankers import TransformersSimilarityRanker
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from utils import read_input_json, serialize_generated_answer

chat_template = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Answer the provided query using the following documents as context while speaking like a pirate:


Context:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}
<eot_id><start_header_id|>user<|end_header_id|>
query: {{query}}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Answer:
"""


generator = LlamaCppGenerator(
    model="models/Meta-Llama-3-8B-Instruct.Q8_0.gguf",
    n_ctx=2048,
    n_batch=512,
    model_kwargs={"n_gpu_layers": -1, "n_predict": -1},
    generation_kwargs={
        "max_tokens": 200,
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.9,
    },
)

text_embedder = SentenceTransformersTextEmbedder(model="hkunlp/instructor-large")

generator.warm_up()
text_embedder.warm_up()

document_store = ChromaDocumentStore(persist_path="chromaDB")


rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component(
    "embedder_retriever",
    ChromaEmbeddingRetriever(document_store=document_store, top_k=5),
)
rag_pipeline.add_component("ranker", TransformersSimilarityRanker())
rag_pipeline.add_component("prompt_builder", PromptBuilder(template=chat_template))
rag_pipeline.add_component("llm", generator)
rag_pipeline.add_component("answer_builder", AnswerBuilder())


rag_pipeline.connect("text_embedder", "embedder_retriever")
rag_pipeline.connect("embedder_retriever", "ranker")
rag_pipeline.connect("ranker", "prompt_builder")
rag_pipeline.connect("prompt_builder", "llm")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
rag_pipeline.connect("llm.meta", "answer_builder.meta")
rag_pipeline.connect("embedder_retriever", "answer_builder.documents")


rag_pipeline.draw("visual_design/rag_pipeline.png")

prompts = read_input_json("documents/input/input.json")
results = []
for prompt_dict in prompts:
    question = prompt_dict["question"]
    staff_answer = prompt_dict["staff_answer"]

    result = rag_pipeline.run(
        {
            "text_embedder": {"text": question},
            "prompt_builder": {"query": question},
            "answer_builder": {"query": question},
            "ranker": {"query": question},
        }
    )
    result_with_original = {"staff_answer": staff_answer, "generated_answer": result}
    results.append(result_with_original)

serialize_generated_answer(results)
