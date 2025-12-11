import json

def load_ground_truth(path):
    """Load ground truth file (JSONL)."""
    gt = {}
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            question = obj["question"]
            gt[question] = obj
    return gt