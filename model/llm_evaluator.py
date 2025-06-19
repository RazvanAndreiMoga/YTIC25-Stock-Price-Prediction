class LLMEvaluator:
    def __init__(self):
        pass

    def evaluate(self, predictions, true_directions):
        correct = 0
        total = len(predictions)
        for pred, true in zip(predictions, true_directions):
            if pred.strip().upper() == true.strip().upper():
                correct += 1
        accuracy = correct / total if total else 0
        print(f"✅ LLM Direction Accuracy: {accuracy:.2%} ({correct}/{total})")
        return accuracy
