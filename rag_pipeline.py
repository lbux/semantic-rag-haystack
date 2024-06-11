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

from utils.serializing import read_input_json, serialize_generated_answer
from utils.structure_documents import create_source_path

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Initialize index pipeline components that require parameters before adding them
document_store = InMemoryDocumentStore()
document_splitter = DocumentSplitter(
    split_by="word", split_length=256, split_overlap=32
)
document_embedder = SentenceTransformersDocumentEmbedder(
    model="BAAI/bge-small-en-v1.5", device=ComponentDevice(device)
)
file_router = FileTypeRouter(
    mime_types=["application/pdf", "text/markdown", "text/plain"]
)

# The following lines add each required component for document processing
# Which we will then store in an in memory document store
index_pipeline = Pipeline()
index_pipeline.add_component("document_splitter", document_splitter)
index_pipeline.add_component("document_embedder", document_embedder)
index_pipeline.add_component("document_writer", DocumentWriter(document_store))
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

# We run the indexing pipeline and provide the source path to index
index_pipeline.run({"file_router": {"sources": create_source_path()}})

# Initialize rag pipeline components that require parameters before adding them
text_embedder = SentenceTransformersTextEmbedder(
    model="BAAI/bge-small-en-v1.5", device=ComponentDevice(device)
)
embedding_retriever = InMemoryEmbeddingRetriever(document_store)
bm25_retriever = InMemoryBM25Retriever(document_store)
ranker = TransformersSimilarityRanker(
    model="BAAI/bge-reranker-base", top_k=4, device=ComponentDevice(device)
)

# LLM specific parameter
# Will offload to GPU by default
# Using q6 quant of LLama3 by default
llm = LlamaCppChatGenerator(
    model="models/Meta-Llama-3-8B-Instruct-Q6_K.gguf",
    model_kwargs={"n_gpu_layers": -1, "n_predict": -1},
)
llm.warm_up()

# The following lines handle the prompt used for generating answers.
# Only the system message should be modified. Do not modify context logic
system_message = ChatMessage.from_system(
    """
    Use the context provided below to generate a concise and accurate answer to the query. 
    If the context does not contain enough information to provide a reliable answer, 
    respond with: "I cannot answer this question with the context provided." and nothing else.

    Example 1:
    Query: I think I received an incorrect grade for my homework. Can I have someone review it again?
    Answer: Provided that 5 days has not passed since the grade was posted, you can request a regrade on EdDiscussion by making a private post.

    Example 2:
    Query: Do you recommend I take CS 161 if I plan to go to grad school?
    Answer: I cannot answer this question with the context provided.

    Context:
    {% for doc in documents %}
    {{ doc.content }}
    {% endfor %};
    """
)
user_message = ChatMessage.from_user("query: {{query}}")
assistent_message = ChatMessage.from_assistant("Answer: ")

chat_template = [system_message, user_message, assistent_message]

# The following lines add each required component retrieval, ranking
# and generation
rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("embedding_retriever", embedding_retriever)
rag_pipeline.add_component("bm25_retriever", bm25_retriever)
rag_pipeline.add_component("document_joiner", DocumentJoiner())
rag_pipeline.add_component("ranker", ranker)
rag_pipeline.add_component("prompt_builder", ChatPromptBuilder(template=chat_template))
rag_pipeline.add_component("llm", llm)
rag_pipeline.add_component("answer_builder", AnswerBuilder())

rag_pipeline.connect("text_embedder", "embedding_retriever")
rag_pipeline.connect("bm25_retriever", "document_joiner")
rag_pipeline.connect("embedding_retriever", "document_joiner")
rag_pipeline.connect("document_joiner", "ranker")
rag_pipeline.connect("ranker", "prompt_builder")
rag_pipeline.connect("prompt_builder", "llm")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
# rag_pipeline.connect("llm.meta", "answer_builder.meta")
rag_pipeline.connect("embedding_retriever", "answer_builder.documents")

rag_pipeline.draw("visual_design/rag_pipeline.png")

# Read in from the input file and run rag pipeline per entry
prompts = read_input_json("documents/input/input.json")
results = []
for prompt_dict in prompts:
    query = prompt_dict["question"]
    answer = prompt_dict["answer"]
    result = rag_pipeline.run(
        {
            "text_embedder": {"text": query},
            "bm25_retriever": {"query": query},
            "prompt_builder": {"query": query},
            "answer_builder": {"query": query},
            "ranker": {"query": query},
        }
    )
    result_with_original = {"answer": answer, "generated_answer": result}
    results.append(result_with_original)

serialize_generated_answer(results)
