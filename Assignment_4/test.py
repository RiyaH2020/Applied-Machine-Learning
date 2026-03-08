# ============================================================
# Name        : Riya Shyam Huddar
# Roll Number : MDS202431
# Course      : Applied Machine Learning
# Assignment  : Assignment 4
# ============================================================

import time
import requests
import joblib
import pytest
import subprocess
import sys


from score import score


# -------------------------------------------------------
# Load model once for unit tests
# -------------------------------------------------------

model = joblib.load("CSVC_best_model.pkl")


# =======================================================
# UNIT TESTS - score()
# =======================================================

def test_smoke():
    result = score("Hello world", model, 0.5)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_output_types():
    prediction, propensity = score("Sample text", model, 0.5)
    assert isinstance(prediction, bool)
    assert isinstance(propensity, float)


def test_probability_range():
    _, propensity = score("See you at 5pm", model, 0.5)
    assert 0.0 <= propensity <= 1.0


def test_deterministic_prediction():
    text = "Win cash prize today"
    p1 = score(text, model, 0.5)
    p2 = score(text, model, 0.5)
    assert p1 == p2


# -------------------------------------------------------
# Threshold Behavior
# -------------------------------------------------------

def test_threshold_zero_always_true():
    prediction, _ = score("Any message", model, 0)
    assert prediction is True


def test_threshold_one_always_false():
    prediction, _ = score("Any message", model, 1)
    assert prediction is False


def test_threshold_equal_probability():
    text = "Limited time offer"
    _, prob = score(text, model, 0.5)
    pred_equal, _ = score(text, model, prob)
    assert pred_equal is True


# -------------------------------------------------------
# Regression Spam / Ham Tests
# -------------------------------------------------------

@pytest.mark.parametrize("spam_text", [
    "Congratulations! You have won a $1000 Walmart gift card. Click here to claim now!",
    "You have been selected for a cash prize!",
    "URGENT! Claim your reward now!",
    "You have won a lottery worth $5000!!!",
    "URGENT! Your account has been compromised. Click now!"
])
def test_multiple_spam_examples(spam_text):
    prediction, _ = score(spam_text, model, 0.5)
    assert prediction is True


@pytest.mark.parametrize("ham_text", [
    "Hi Riya, are we meeting tomorrow at 10am for the project discussion?",
    "Are we still meeting tomorrow?",
    "Please send me the report by evening.",
    "Call me when you reach home.",
    "Happy birthday! Have a great year ahead."
])
def test_multiple_ham_examples(ham_text):
    prediction, _ = score(ham_text, model, 0.5)
    assert prediction is False


# -------------------------------------------------------
# Edge Cases
# -------------------------------------------------------

def test_empty_string():
    prediction, propensity = score("", model, 0.5)
    assert isinstance(prediction, bool)
    assert 0.0 <= propensity <= 1.0


def test_whitespace_input():
    prediction, propensity = score("     ", model, 0.5)
    assert isinstance(prediction, bool)
    assert 0.0 <= propensity <= 1.0


def test_long_input():
    long_text = "Free offer " * 1000
    prediction, propensity = score(long_text, model, 0.5)
    assert isinstance(prediction, bool)
    assert 0.0 <= propensity <= 1.0


# -------------------------------------------------------
# Validation Tests
# -------------------------------------------------------

def test_invalid_text_type():
    with pytest.raises(TypeError):
        score(123, model, 0.5)


def test_invalid_threshold_type():
    with pytest.raises(TypeError):
        score("hello", model, "0.5")


def test_threshold_out_of_bounds_low():
    with pytest.raises(ValueError):
        score("hello", model, -0.1)


def test_threshold_out_of_bounds_high():
    with pytest.raises(ValueError):
        score("hello", model, 1.5)


def test_model_none():
    with pytest.raises(ValueError):
        score("hello", None, 0.5)


# -------------------------------------------------------
# Defensive Branch Coverage Tests
# -------------------------------------------------------

class DummyNoProba:
    pass


def test_model_without_predict_proba():
    with pytest.raises(AttributeError):
        score("hello", DummyNoProba(), 0.5)


class DummyNoClasses:
    def predict_proba(self, X):
        return [[0.4, 0.6]]


def test_model_without_classes():
    with pytest.raises(AttributeError):
        score("hello", DummyNoClasses(), 0.5)


class DummyWrongClasses:
    classes_ = [0, 2]

    def predict_proba(self, X):
        return [[0.4, 0.6]]


def test_positive_class_missing():
    with pytest.raises(ValueError):
        score("hello", DummyWrongClasses(), 0.5)


# =======================================================
# INTEGRATION TEST - Launch Flask via Command Line
# =======================================================

def test_flask_integration():

    process = subprocess.Popen(
        [sys.executable, "app.py"]
    )

    time.sleep(5)

    try:
        for _ in range(5):
            try:
                response = requests.post(
                    "http://127.0.0.1:5000/score",
                    json={"text": "Free lottery! Claim now!"}
                )
                break
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        else:
            pytest.fail("Flask server did not start.")

        assert response.status_code == 200

        data = response.json()
        assert "prediction" in data
        assert "propensity" in data
        assert isinstance(data["prediction"], bool)
        assert isinstance(data["propensity"], float)
        assert 0.0 <= data["propensity"] <= 1.0

    finally:
        process.terminate()
        process.wait()


# =======================================================
# Flask Test Client - Coverage
# =======================================================

@pytest.fixture
def client():
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_homepage(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Spam Classifier" in response.data


def test_score_endpoint_json(client):
    response = client.post(
        "/score",
        json={"text": "Free lottery! Claim now!"}
    )
    assert response.status_code == 200


def test_missing_text_json(client):
    response = client.post("/score", json={})
    assert response.status_code == 400


def test_missing_text_form(client):
    response = client.post("/score", data={})
    assert response.status_code == 400


def test_invalid_text_type_json(client):
    response = client.post("/score", json={"text": 123})
    assert response.status_code == 400



# =======================================================
# DOCKER TEST
# =======================================================

# ------------------------------------------------------------------
# Docker configuration
# ------------------------------------------------------------------
DOCKER_IMAGE = "aml-flask-app"
DOCKER_CONTAINER = "test_container"
DOCKER_HOST_PORT = 5000
DOCKER_BASE_URL = f"http://127.0.0.1:{DOCKER_HOST_PORT}"


def _docker_cmd(cmd):
    """Run a docker CLI command."""
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def _wait_for_container(url, retries=10, delay=1):
    """Wait until the container endpoint becomes reachable."""
    for _ in range(retries):
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(delay)
    return False


def test_docker():
    """
    Docker integration test:
      1. Build docker image
      2. Run container
      3. Send request to /score endpoint
      4. Validate response
      5. Stop and remove container
    """

    # --------------------------------------------------------------
    # Step 0: Remove any leftover container
    # --------------------------------------------------------------
    _docker_cmd(f"docker rm -f {DOCKER_CONTAINER}")

    # --------------------------------------------------------------
    # Step 1: Build image
    # --------------------------------------------------------------
    build = _docker_cmd(f"docker build -t {DOCKER_IMAGE} .")
    assert build.returncode == 0, f"Docker build failed:\n{build.stderr}"

    # --------------------------------------------------------------
    # Step 2: Run container
    # --------------------------------------------------------------
    run = _docker_cmd(
        f"docker run -d -p {DOCKER_HOST_PORT}:5000 --name {DOCKER_CONTAINER} {DOCKER_IMAGE}"
    )
    assert run.returncode == 0, f"Docker run failed:\n{run.stderr}"

    try:

        # ----------------------------------------------------------
        # Step 3: Wait for Flask app to start
        # ----------------------------------------------------------
        assert _wait_for_container(DOCKER_BASE_URL), "Container did not start in time"

        # ----------------------------------------------------------
        # Step 4: Send scoring request
        # ----------------------------------------------------------
        payload = {"text": "Congratulations you won a free prize"}

        response = requests.post(
            f"{DOCKER_BASE_URL}/score",
            json=payload
        )

        assert response.status_code == 200

        data = response.json()

        assert "prediction" in data
        assert "propensity" in data

        assert isinstance(data["prediction"], bool)
        assert isinstance(data["propensity"], float)

    finally:

        # ----------------------------------------------------------
        # Step 5: Cleanup container
        # ----------------------------------------------------------
        _docker_cmd(f"docker stop {DOCKER_CONTAINER}")
        _docker_cmd(f"docker rm {DOCKER_CONTAINER}")