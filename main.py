from haystack import Pipeline
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.rankers import TransformersSimilarityRanker
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

from utils import read_input_json, serialize_generated_answer

chat_template = """
You are a teaching assistant for a course on Data Management.
You will be provided with queries from students that have been posted on a discussion board, EdDiscussion.
You will answer the queries using the data in the syllabus.
If you can not find the answer only using the syllabus, do not provide an answer.

Here is an example of the expected answer based on a query:

QUERY: "Are we able to use private edStem posts to contact you or is email preferred?"
ANSWER: "Yes, you can use private edStem posts or email to contact us. Also, there is a section in the syllabus describing how/where to contact us that you may find helpful."

Context:
{% for doc in documents %}
  {{ doc.content }}
{% endfor %}
query: {{query}}
Answer:
"""

generator = LlamaCppGenerator(
    model="models/WizardLM-2-7B.Q8_0.gguf",
    n_ctx=12000,
    model_kwargs={"n_gpu_layers": -1},
    generation_kwargs={
        # i pulled these from a random reddit post lol
        # but they seem to work well?
        "max_tokens": 128,
        "temperature": 0.7,
        "repeat_penalty": 1.176,
        "top_k": 40,
        "top_p": 0.1,
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


rag_pipeline.draw("rag_pipeline.png")

prompts = read_input_json("spring22.json")
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
