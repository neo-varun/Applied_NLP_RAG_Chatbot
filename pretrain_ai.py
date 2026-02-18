from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def classify_review(text):
    labels = [
        "Battery Issue",
        "Screen Quality",
        "Delivery Speed",
        "Build Quality",
        "Performance",
        "Value for Money",
    ]

    result = classifier(text, labels)
    return result["labels"][0], result["scores"][0]
