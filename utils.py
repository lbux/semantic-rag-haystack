from dataclasses import asdict
import json


def serialize_pipeline_results(results):
    serialized_data = []
    for result in results:
        answers = result.get("answer_builder", {}).get("answers", [])

        for answer in answers:
            answer_dict = asdict(answer)
            if "documents" in answer_dict:
                answer_dict["documents"] = [asdict(doc) for doc in answer.documents]
            serialized_data.append(answer_dict)

    with open("serialized_data.json", "w", encoding="utf-8") as f:
        json.dump(serialized_data, f, ensure_ascii=False, indent=2)


def read_serialized_data():
    queries = []
    documents = []
    answers = []

    with open("serialized_data.json", "r") as f:
        serialized_data = json.load(f)
        for entry in serialized_data:
            query = entry.get("query", "")
            queries.append(query)

            docs = [
                doc["content"] for doc in entry.get("documents", []) if "content" in doc
            ]
            documents.append(docs)

            answer_data = entry.get("data", "")
            answers.append(answer_data)
    return queries, documents, answers
