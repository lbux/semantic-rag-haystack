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
        "questions",
        "responses",
        "ground_truths",
    ],
}


def serialize_generated_answer(results):
    serialized_data = []

    for result in results:
        staff_answer = result["staff_answer"]
        generated_answers = result["generated_answer"].get("answer_builder", {}).get("answers", [])

        for generated_answer in generated_answers:
            answer_dict = asdict(generated_answer)
            answer_dict["staff_answer"] = staff_answer 
            if "documents" in answer_dict:
                answer_dict["documents"] = [asdict(doc) for doc in generated_answer.documents]
            serialized_data.append(answer_dict)

    with open("serialized_generated_data.json", "w", encoding="utf-8") as f:
        json.dump(serialized_data, f, ensure_ascii=False, indent=2)



def read_serialized_generated_answer():
    queries = []
    documents = []
    answers = []
    staff_answers = []

    with open("serialized_generated_data.json", "r", encoding="utf-8") as f:
        serialized_data = json.load(f)
        for entry in serialized_data:
            query = entry.get("query", "")
            queries.append(query)

            document = [
                doc["content"] for doc in entry.get("documents", []) if "content" in doc
            ]
            documents.append(document)

            answer = entry.get("data", "")
            answers.append(answer)
            staff_answer = entry.get("staff_answer", "")
            staff_answers.append(staff_answer)
    return queries, documents, answers, staff_answers


def metric_to_params(metric, data):
    params = {}
    for param in uptrain_parameters_mapping.get(metric, []):
        if param in data and data[param]:
            params[param] = data[param]
    return {"evaluator": params}


def serialize_evaluation_results(evaluation_results):
    with open("serialized_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

def read_input_json(file_name):
    qa_pairs = []

    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
        for entry in data:
            if entry.get("class", "") == "Course Policy/Format":
                question = entry.get("text", "").strip()
                if not question:
                    continue

                admin_answer, staff_answer = None, None
                for answer in entry.get("answers", []):
                    if answer["user"]["role"] == "admin" and not admin_answer:
                        admin_answer = answer["text"].strip()
                        break
                    elif answer["user"]["role"] == "staff" and not staff_answer:
                        staff_answer = answer["text"].strip()

                staff_answer = admin_answer if admin_answer else staff_answer

                qa_pairs.append({"question": question, "staff_answer": staff_answer})

    return qa_pairs

