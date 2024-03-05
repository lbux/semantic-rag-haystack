from dataclasses import asdict
import json

from haystack_integrations.components.evaluators.uptrain import UpTrainMetric

uptrain_parameters_mapping = {
    UpTrainMetric.CONTEXT_RELEVANCE: ["questions", "contexts"],
    UpTrainMetric.FACTUAL_ACCURACY: ["questions", "contexts", "responses"],
    UpTrainMetric.RESPONSE_RELEVANCE: ["questions", "responses"],
    UpTrainMetric.RESPONSE_COMPLETENESS: ["questions", "responses"],
    UpTrainMetric.RESPONSE_COMPLETENESS_WRT_CONTEXT: [
        "questions",
        "contexts",
        "responses",
    ],
    UpTrainMetric.RESPONSE_CONSISTENCY: ["questions", "contexts", "responses"],
    UpTrainMetric.RESPONSE_CONCISENESS: ["questions", "responses"],
    UpTrainMetric.CRITIQUE_LANGUAGE: ["responses"],
    UpTrainMetric.CRITIQUE_TONE: ["responses"],
    UpTrainMetric.GUIDELINE_ADHERENCE: [
        "questions",
        "responses",
    ],
    UpTrainMetric.RESPONSE_MATCHING: [
        "responses",
        "ground_truths",
    ],
}


def serialize_generated_answer(results):
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


def read_serialized_generated_answer():
    queries = []
    documents = []
    answers = []

    with open("serialized_data.json", "r", encoding="utf-8") as f:
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


def metric_to_params(metric, data):
    params = {}
    for param in uptrain_parameters_mapping.get(metric, []):
        if param in data and data[param]:
            params[param] = data[param]
    return {"evaluator": params}


def serialize_evaluation_results(evaluation_results):
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
