# ============================================================
# Name        : Riya Shyam Huddar
# Roll Number : MDS202431
# Course      : Applied Machine Learning
# Assignment  : Assignment 3
# ============================================================

from typing import Tuple


def score(text: str,
          model,
          threshold: float = 0.5) -> Tuple[bool, float]:
    """
    Score a trained sklearn model on input text.

    Args:
        text (str): Raw input message.
        model: Trained sklearn pipeline.
        threshold (float): Decision threshold in [0,1].

    Returns:
        Tuple[bool, float]:
            prediction (bool): True if positive class predicted.
            propensity (float): Predicted probability of positive class.
    """

    # ----------------------------
    # Input validation
    # ----------------------------
    if not isinstance(text, str):
        raise TypeError(f"'text' must be str, got {type(text).__name__}")

    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("'threshold' must be numeric")

    if not 0 <= threshold <= 1:
        raise ValueError("'threshold' must be between 0 and 1")

    if model is None:
        raise ValueError("model must not be None")

    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model must implement predict_proba()")

    if not hasattr(model, "classes_"):
        raise AttributeError("Model must expose 'classes_' attribute")

    # ----------------------------
    # Predict probability
    # ----------------------------
    probabilities = model.predict_proba([text])

    # Ensure positive class (1) exists
    if 1 not in model.classes_:
        raise ValueError("Positive class label '1' not found in model.classes_")

    positive_index = list(model.classes_).index(1)
    proba = float(probabilities[0][positive_index])

    # ----------------------------
    # Apply threshold
    # ----------------------------
    prediction = bool(proba >= threshold)

    return prediction, proba