# pipeline/classifier.py
"""
Wraps the trained BBMP sklearn classifier.
Returns predicted category + confidence score (max class probability).
"""

import pickle
from dataclasses import dataclass


@dataclass
class Prediction:
    category:   str
    confidence: float   # 0–1


class BBMPClassifier:
    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            package = pickle.load(f)

        self.vectorizer = package["vectorizer"]
        self.clf        = package["classifier"]

        # Not every sklearn estimator exposes predict_proba
        self._has_proba = hasattr(self.clf, "predict_proba")

    def predict(self, text: str) -> Prediction:
        """
        Classify one English text string.

        Parameters
        ----------
        text : str   already-translated English complaint text

        Returns
        -------
        Prediction(category, confidence)
        """
        if not text.strip():
            return Prediction(category="Unknown", confidence=0.0)

        try:
            X        = self.vectorizer.transform([text])
            category = self.clf.predict(X)[0]

            if self._has_proba:
                proba      = self.clf.predict_proba(X)[0]
                confidence = round(float(proba.max()), 4)
            else:
                # Decision-function fallback (e.g. LinearSVC)
                scores     = self.clf.decision_function(X)[0]
                # Softmax-normalise
                import numpy as np
                e = np.exp(scores - scores.max())
                confidence = round(float(e.max() / e.sum()), 4)

            return Prediction(category=category, confidence=confidence)

        except Exception as exc:
            raise ClassificationError(str(exc)) from exc


class ClassificationError(RuntimeError):
    pass