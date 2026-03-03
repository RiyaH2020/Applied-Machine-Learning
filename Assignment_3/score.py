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
        (prediction: bool, propensity: float)
    """

    # ----------------------------
    # Input validation
    # ----------------------------
    if not isinstance(text, str):
        raise TypeError(f"'text' must be str, got {type(text).__name__}")

    if not isinstance(threshold, (int, float)):
        raise TypeError("'threshold' must be numeric")

    if not 0 <= threshold <= 1:
        raise ValueError("'threshold' must be between 0 and 1")

    if model is None:
        raise ValueError("model must not be None")

    # ----------------------------
    # Predict probability
    # ----------------------------
    proba = float(model.predict_proba([text])[0][1])

    # ----------------------------
    # Explicit threshold handling
    # ----------------------------
    if threshold == 0:
        prediction = True
    elif threshold == 1:
        prediction = False
    else:
        prediction = bool(proba >= threshold)

    return prediction, proba