import json
import pickle
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


CSV_PATH = Path("data/BBMP_cleaned.csv")
MODEL_PATH = Path("Models/model_bbmp.pkl")
OUTPUT_DIR = Path("outputs")

TEXT_COLUMN = "Sub Category"
CATEGORY_COLUMN = "Category"


def clean_text(text: str) -> str:
    value = str(text).lower().strip()
    value = re.sub(r"[^\w\s]", " ", value)
    return re.sub(r"\s+", " ", value)


def normalize_category(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


category_map = {
    "electrical": "Street Light",
    "solid waste (garbage) related": "Garbage / Sanitation",
    "road maintenance(engg)": "Road Repair",
    "road infrastructure": "Road Repair",
    "storm water drain(swd)": "Drainage / SWD",
    "sanitation": "Garbage / Sanitation",
    "health dept": "Health / Sanitation",
    "water crisis": "Water Supply",
    "parks and play grounds": "Parks",
    "forest": "Parks / Forest",
    "town planning": "Town Planning",
    "revenue department": "Revenue",
    "veterinary": "Veterinary",
    "advertisement": "Advertisement",
    "others": "Others",
}


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=[TEXT_COLUMN, CATEGORY_COLUMN]).copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)
    df["category_normalized"] = df[CATEGORY_COLUMN].apply(normalize_category)
    df["target"] = df["category_normalized"].map(category_map).fillna("Others")

    _, x_test, _, y_test = train_test_split(
        df[TEXT_COLUMN],
        df["target"],
        test_size=0.20,
        random_state=42,
        stratify=df["target"],
    )

    with MODEL_PATH.open("rb") as model_file:
        package = pickle.load(model_file)

    vectorizer = package["vectorizer"]
    classifier = package["classifier"]

    x_test_vec = vectorizer.transform(x_test)
    y_pred = classifier.predict(x_test_vec)

    labels = [str(label) for label in classifier.classes_]
    accuracy = float(accuracy_score(y_test, y_pred))

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()

    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "model_classification_report.csv"
    matrix_path = OUTPUT_DIR / "model_accuracy_matrix.csv"
    summary_path = OUTPUT_DIR / "model_metrics_summary.json"

    report_df.to_csv(report_path)
    matrix_df.to_csv(matrix_path)

    off_diagonal = []
    for row_index, actual in enumerate(labels):
        for col_index, predicted in enumerate(labels):
            if actual == predicted:
                continue
            count = int(matrix[row_index, col_index])
            if count > 0:
                off_diagonal.append(
                    {
                        "actual": actual,
                        "predicted": predicted,
                        "count": count,
                    }
                )
    off_diagonal.sort(key=lambda item: item["count"], reverse=True)

    summary = {
        "model_path": str(MODEL_PATH),
        "dataset_path": str(CSV_PATH),
        "test_size": int(len(y_test)),
        "accuracy": round(accuracy, 4),
        "labels": labels,
        "top_confusions": off_diagonal[:12],
        "report_csv": str(report_path),
        "confusion_matrix_csv": str(matrix_path),
    }

    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
