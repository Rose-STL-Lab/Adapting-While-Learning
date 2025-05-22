import json

with open("pde_test.json", "r") as f:
    data = json.load(f)

cnt = 0
for item in data:
    try:
        answer = json.loads(item["processed"][-1]["function_call"]["arguments"])["answer"]
        if answer == item["correct"]:
            cnt +=1
            item["correct"] = True
        else:
            item["correct"] = False
    except Exception as e:
        print(e)
        pass


print(cnt/len(data))