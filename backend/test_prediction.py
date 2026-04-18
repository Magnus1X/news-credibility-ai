"""
test_prediction.py — Quick sanity check for the news credibility model.

Run from the backend/ directory:
    python3 test_prediction.py

Or from project root:
    python3 backend/test_prediction.py
"""

import os
import sys
import joblib

# Allow running from project root OR backend/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from preprocessing import preprocess_text

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
X_TEST_PATH = os.path.join(BASE_DIR, "..", "artifacts", "X_test.pkl")
Y_TEST_PATH = os.path.join(BASE_DIR, "..", "artifacts", "y_test.pkl")

print("Loading model and vectorizer...")
model = joblib.load(MODEL_PATH)
vec = joblib.load(VECTORIZER_PATH)
print(f"  Model: {type(model).__name__}")
print(f"  Vectorizer: {type(vec).__name__} | vocab size: {len(vec.vocabulary_)}")

# ─── Test 1: Benchmark on saved test set ─────────────────────────────────────
print("\n─── Test 1: Accuracy on saved X_test ───")
X_test = joblib.load(X_TEST_PATH)
y_test = joblib.load(Y_TEST_PATH)

X_vec = vec.transform(X_test)
preds = model.predict(X_vec)
acc = sum(p == t for p, t in zip(preds, y_test)) / len(y_test)
print(f"  Accuracy: {acc:.2%} on {len(y_test)} samples")
assert acc > 0.90, f"Accuracy too low! Expected >90%, got {acc:.2%}"
print("  ✅ PASSED")

# ─── Test 2: Real-world fake news ────────────────────────────────────────────
print("\n─── Test 2: Real-world fake news snippet ───")
FAKE_TEXT = """
BREAKING: Scientists confirm the moon is made of cheese and NASA has been hiding it
since 1969. Sources inside the deep state reveal that world governments have suppressed
this information for decades to maintain control over the population. Trump and Biden
are secretly the same person — a clone created by lizard people in 1947.
"""
cleaned = preprocess_text(FAKE_TEXT)
vector = vec.transform([cleaned])
pred = model.predict(vector)[0]
prob = model.predict_proba(vector)[0]
label = {0: "Real News", 1: "Fake News"}.get(int(pred), str(pred))
print(f"  Prediction : {label}")
print(f"  Confidence : {max(prob) * 100:.1f}%")
print(f"  ✅ PASSED" if pred == 1 else "  ⚠️  Expected Fake News")

# ─── Test 3: Real news snippet ───────────────────────────────────────────────
print("\n─── Test 3: Real news snippet ───")
REAL_TEXT = """
The Federal Reserve raised interest rates by a quarter percentage point on Wednesday,
marking its tenth rate hike in just over a year, as the central bank continues its
effort to bring down inflation. Fed Chair Jerome Powell said that the committee would
need to see more data before deciding whether to pause or continue raising rates.
"""
cleaned = preprocess_text(REAL_TEXT)
vector = vec.transform([cleaned])
pred = model.predict(vector)[0]
prob = model.predict_proba(vector)[0]
label = {0: "Real News", 1: "Fake News"}.get(int(pred), str(pred))
print(f"  Prediction : {label}")
print(f"  Confidence : {max(prob) * 100:.1f}%")
print(f"  ✅ PASSED" if pred == 0 else "  ⚠️  Expected Real News")

print("\n✅ All tests complete.")
