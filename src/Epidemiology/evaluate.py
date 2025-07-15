from jload import jload
import json

data = jload("test.json")

models = [""]

for model in models:
    a = 0
    b = 0
    c = 0
    dd = 0
    pns = 0
    pis = 0
    for d in data:
        try:
            pn_correct = f"the answer is {d['correct_option']}".lower() in d[model + "pn"].lower()
            pi_correct = json.loads(d[model][-1]["content"])["parameters"]["answer"].lower() == d['correct_option'].lower() if d[model] else 0
            tool = len(d[model]) > 3 if d[model] else 0
            pns += pn_correct
            pis += pi_correct
            if not tool and pn_correct:
                a += 1
            elif tool and pn_correct:
                b += 1
            elif tool and not pn_correct:
                c += 1
            elif not tool and not pn_correct:
                dd += 1
        except Exception as e:
            pass
    print(f"Model: {model}, pn_correct: {pns}, pi_correct: {pis}, a: {a}, b: {b}, c: {c}, d: {dd}, total: {len(data)}")
    print(f"pn_correct: {pns / len(data)}, pi_correct: {pis / len(data)}")
    print(0.5 * a / (a + b) + 0.5 * c / (c + dd))
