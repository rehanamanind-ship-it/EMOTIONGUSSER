import json
import importlib
import importlib.util
from difflib import SequenceMatcher
from pathlib import Path

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


DATASET_PATH = Path(__file__).with_name("DATASET.JSON")


def normalize_pattern(text):
    return text.strip().casefold()


def empty_dataset():
    return {"examples": []}


def save_dataset(dataset):
    DATASET_PATH.write_text(json.dumps(dataset, indent=2), encoding="utf-8")


def load_dataset():
    if not DATASET_PATH.exists() or not DATASET_PATH.read_text(encoding="utf-8").strip():
        dataset = empty_dataset()
        save_dataset(dataset)
        return dataset

    try:
        with DATASET_PATH.open("r", encoding="utf-8") as file:
            raw_data = json.load(file)
    except json.JSONDecodeError:
        print("DATASET.JSON is invalid. Starting with an empty dataset.")
        dataset = empty_dataset()
        save_dataset(dataset)
        return dataset

    if isinstance(raw_data, list):
        dataset = {"examples": raw_data}
    elif isinstance(raw_data, dict):
        dataset = {"examples": raw_data.get("examples", [])}
    else:
        dataset = empty_dataset()

    cleaned_examples = []
    for item in dataset["examples"]:
        if not isinstance(item, dict):
            continue
        pattern = str(item.get("pattern", "")).strip()
        emotion = str(item.get("emotion", "")).strip()
        if pattern and emotion:
            cleaned_examples.append({"pattern": pattern, "emotion": emotion})

    dataset["examples"] = cleaned_examples
    save_dataset(dataset)
    return dataset


def build_lookup(examples):
    return {normalize_pattern(item["pattern"]): item["emotion"] for item in examples}


def train_model(examples):
    if len(examples) < 2:
        return None

    try:
        CountVectorizer = importlib.import_module(
            "sklearn.feature_extraction.text"
        ).CountVectorizer
        MultinomialNB = importlib.import_module("sklearn.naive_bayes").MultinomialNB
        Pipeline = importlib.import_module("sklearn.pipeline").Pipeline
    except Exception:
        return None

    patterns = [normalize_pattern(item["pattern"]) for item in examples]
    emotions = [item["emotion"] for item in examples]

    model = Pipeline(
        [
            ("vectorizer", CountVectorizer(analyzer="char", ngram_range=(1, 3))),
            ("classifier", MultinomialNB()),
        ]
    )
    model.fit(patterns, emotions)
    return model


def heuristic_guess(pattern):
    lowered = normalize_pattern(pattern)

    if "<3" in lowered or "xoxo" in lowered:
        return "love", 0.82, "heuristic"
    if "uwu" in lowered or "owo" in lowered or ":3" in lowered or "^_^" in lowered:
        return "cuteness", 0.80, "heuristic"
    if "t_t" in lowered or "q_q" in lowered or ";_;" in lowered or ":'(" in lowered:
        return "sadness", 0.84, "heuristic"
    if "grr" in lowered or ">:(" in lowered or "d:<" in lowered:
        return "anger", 0.84, "heuristic"
    if "!" in pattern and "?" not in pattern:
        return "excitement", 0.68, "heuristic"
    if "..." in pattern:
        return "sadness", 0.66, "heuristic"

    return None, 0.0, None


def ml_guess(pattern, model):
    if model is None:
        return None, 0.0

    normalized = normalize_pattern(pattern)
    probabilities = model.predict_proba([normalized])[0]
    best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    emotion = model.classes_[best_index]
    confidence = float(probabilities[best_index])
    return emotion, confidence


def predict_emotion(pattern, examples, model):
    normalized = normalize_pattern(pattern)
    lookup = build_lookup(examples)

    if normalized in lookup:
        return {
            "emotion": lookup[normalized],
            "confidence": 1.0,
            "matched_pattern": pattern.strip(),
            "method": "exact",
            "is_new": False,
        }

    ml_emotion, ml_confidence = ml_guess(pattern, model)
    if ml_emotion and ml_confidence >= 0.45:
        return {
            "emotion": ml_emotion,
            "confidence": ml_confidence,
            "matched_pattern": None,
            "method": "machine-learning",
            "is_new": True,
        }

    best_match = None
    best_score = 0.0
    for item in examples:
        score = SequenceMatcher(None, normalized, normalize_pattern(item["pattern"])).ratio()
        if score > best_score:
            best_score = score
            best_match = item

    if best_match and best_score >= 0.62:
        return {
            "emotion": best_match["emotion"],
            "confidence": best_score,
            "matched_pattern": best_match["pattern"],
            "method": "fuzzy",
            "is_new": True,
        }

    emotion, confidence, method = heuristic_guess(pattern)
    if emotion:
        return {
            "emotion": emotion,
            "confidence": confidence,
            "matched_pattern": None,
            "method": method,
            "is_new": True,
        }

    return {
        "emotion": None,
        "confidence": 0.0,
        "matched_pattern": None,
        "method": "unknown",
        "is_new": True,
    }


def ask_yes_no(prompt):
    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            print("\nInput ended, treating this as 'no'.")
            return False
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer with yes or no.")


def add_new_example(pattern, suggested_emotion, dataset):
    print("\nThis pattern is new to the dataset.")
    if not ask_yes_no("Do you want to add its meaning to DATASET.JSON? (yes/no): "):
        return dataset

    if suggested_emotion:
        try:
            emotion = input(
                f"Press Enter to keep '{suggested_emotion}', or type the correct emotion: "
            ).strip()
        except EOFError:
            print("\nInput ended, keeping the suggested emotion.")
            emotion = suggested_emotion
        if not emotion:
            emotion = suggested_emotion
    else:
        try:
            emotion = input("What emotion should I save for this pattern? ").strip()
        except EOFError:
            print("\nInput ended, so this pattern was not saved.")
            return dataset

    if not emotion:
        print("Nothing was saved because no emotion was provided.")
        return dataset

    dataset["examples"].append({"pattern": pattern.strip(), "emotion": emotion})
    save_dataset(dataset)
    print(f"Saved '{pattern.strip()}' as '{emotion}' in DATASET.JSON.")
    return dataset


def show_examples(examples):
    if not examples:
        print("Examples: no training examples found in DATASET.JSON yet.")
        return
    sample = ", ".join(item["pattern"] for item in examples[:12])
    print("Examples:", sample)


def main():
    dataset = load_dataset()
    model = train_model(dataset["examples"])

    print("Emotion Identifier")
    print("Type a face made of punctuation or letters, and I will guess the emotion.")
    print("Type 'exit' anytime to stop.\n")
    if SKLEARN_AVAILABLE:
        print("Learning mode: scikit-learn model is active.")
    else:
        print("Learning mode: scikit-learn is not installed, so fallback matching is active.")
    show_examples(dataset["examples"])

    while True:
        try:
            user_input = input("\nEnter a pattern: ").strip()
        except EOFError:
            print("\nInput ended. Goodbye.")
            break
        if not user_input:
            print("Please type a pattern like :) or T_T.")
            continue

        if user_input.casefold() in {"exit", "quit"}:
            print("Goodbye.")
            break

        result = predict_emotion(user_input, dataset["examples"], model)

        if result["emotion"]:
            print(f"\nPredicted emotion: {result['emotion']}")
            print(f"Confidence: {result['confidence']:.0%}")
            if result["method"] == "machine-learning":
                print("This was guessed by the learning model trained from DATASET.JSON.")
            elif result["method"] == "fuzzy" and result["matched_pattern"]:
                print(f"Closest known pattern: {result['matched_pattern']}")
            elif result["method"] == "heuristic":
                print("This was guessed using simple pattern rules.")
        else:
            print("\nI do not know this pattern yet.")

        if result["is_new"]:
            dataset = add_new_example(user_input, result["emotion"], dataset)
            model = train_model(dataset["examples"])


if __name__ == "__main__":
    main()
