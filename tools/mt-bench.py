import json
from collections import defaultdict
from tabulate import tabulate
import argparse
import openpyxl
import os

# Load questions into a dict: question_id -> category
def load_questions(path):
    id_to_category = {}
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            id_to_category[item["question_id"]] = item["category"]
    return id_to_category

# Load answers and organize scores per model and category
def load_answers(path, id_to_category):
    scores = defaultdict(lambda: defaultdict(list))  # model -> category -> list of scores
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            qid = item["question_id"]
            model = item["model"]
            score = item["score"]
            category = id_to_category.get(qid)
            if category is not None:
                scores[model][category].append(score)
    return scores

# Compute mean scores
def compute_means(scores):
    table = []
    all_categories = sorted({cat for model_scores in scores.values() for cat in model_scores})
    for model, cat_scores in scores.items():
        row = [model]
        total_scores = []
        for cat in all_categories:
            if cat in cat_scores:
                avg = sum(cat_scores[cat]) / len(cat_scores[cat])
                row.append(f"{avg:.2f}")
                total_scores.extend(cat_scores[cat])
            else:
                row.append("-")
        overall = sum(total_scores) / len(total_scores) if total_scores else 0.0
        row.append(f"{overall:.2f}")
        table.append(row)
    headers = ["Model"] + all_categories + ["Average"]
    return headers, table


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--judgement", type=str, default="FastChat/fastchat/llm_judge/data/mt_bench/model_judgment/gpt-4_single.jsonl", required=False)
parser.add_argument("--question", type=str, default="FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl", required=False)
parser.add_argument("--file_name", type=str, default="MT-Bench-results")
args = parser.parse_args()

# Load and process data
id_to_category = load_questions(args.question)
scores = load_answers(args.judgement, id_to_category)
headers, table = compute_means(scores)

# Print to terminal
print(tabulate(table, headers=headers, tablefmt="grid"))

# Save to Excel
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Metrics"
ws.append(headers)
for row in table:
    ws.append(row)
output_path = os.path.join("./", f"{args.file_name}_metrics.xlsx")
wb.save(output_path)
print(f"✅ Excel file saved to {output_path}")
