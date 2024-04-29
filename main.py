from haystack import Pipeline
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.rankers import TransformersSimilarityRanker
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from utils import read_input_json, serialize_generated_answer


#Gradient Embedder
import os 
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceTGIGenerator
from gradient_haystack.embedders.gradient_text_embedder import GradientTextEmbedder


document_store = InMemoryDocumentStore()
prompt_gradient = """ Answer the query, based on the
content in the documents.

Documents:
{% for doc in documents %}
  {{doc.content}}
{% endfor %}

Query: {{query}}
"""

os.environ["GRADIENT_ACCESS_TOKEN"]="YOUR_GRADIENT_ACCESS_TOKEN"
os.environ["GRADIENT_WORKSPACE_ID"]="GRADIENT_WORKSPACE_ID"

text_embedder_gradient = GradientDocumentEmbedder()
retriever_gradient = InMemoryEmbeddingRetriever(document_store=document_store)
prompt_builder = PromptBuilder(template=prompt_gradient)
generator_HuggingFaceTGI = HuggingFaceTGIGenerator(model="mistralai/Mistral-7B-v0.1", 
									                token="YOUR_HUGGINGFACE_TOKEN")
generator_HuggingFaceTGI.warm_up()



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


generator_Llama = LlamaCppGenerator(
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

text_embedder_STTE = SentenceTransformersTextEmbedder(model="hkunlp/instructor-large")

generator_Llama.warm_up()
text_embedder_STTE.warm_up()

document_store = ChromaDocumentStore(persist_path="chromaDB")


rag_pipeline = Pipeline()

"""
For Gradient
rag_pipeline.add_component(instance=text_embedder, name="text_embedder")
rag_pipeline.add_component(instance=retriever, name="retriever")
rag_pipeline.add_component(instance=prompt_builder, name="prompt_builder")
rag_pipeline.add_component(instance=generator, name="generator")

rag_pipeline.connect("text_embedder", "retriever")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "generator")
"""

rag_pipeline.add_component("text_embedder", text_embedder_STTE)
rag_pipeline.add_component(
    "embedder_retriever",
    ChromaEmbeddingRetriever(document_store=document_store, top_k=5),
)
rag_pipeline.add_component("ranker", TransformersSimilarityRanker())
rag_pipeline.add_component("prompt_builder", PromptBuilder(template=chat_template))
rag_pipeline.add_component("llm", generator_Llama)
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
