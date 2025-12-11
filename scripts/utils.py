import json

def load_jsonl(path):
    lines = []
    with open(path, "r") as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

def save_jsonl(path, list_of_dicts):
    with open(path, "w") as f:
        for d in list_of_dicts:
            f.write(json.dumps(d) + "\n")