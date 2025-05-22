import json
import glob
import random
import os

json_files = glob.glob("make_mujoco/*_questions.json")

all_questions = []

for file in json_files:
    if "sample" in file:
        continue
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        question_type = os.path.basename(file).split('_questions.json')[0]
        for question in data:
            question['type'] = question_type
        all_questions.extend(data)

if len(all_questions) >= 8000:
    sampled_questions = random.sample(all_questions, 8000)
else:
    sampled_questions = all_questions

with open('sampled_questions.json', 'w', encoding='utf-8') as f:
    json.dump(sampled_questions, f, ensure_ascii=False, indent=4)