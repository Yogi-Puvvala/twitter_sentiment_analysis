import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Loading artifacts 
with open(os.path.join(BASE_DIR, "models/le.pkl"), "rb") as f:
    le = pickle.load(f)
with open(os.path.join(BASE_DIR, "models/tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

model = load_model(os.path.join(BASE_DIR, "models/model.keras"))

print("Model and artifacts loaded successfully!")

# Predict function 
def predict(tweet: str) -> dict:
    seq     = tokenizer.texts_to_sequences([tweet])
    pad_seq = pad_sequences(seq,
                             maxlen   = 100,
                             padding  = "post")

    prediction       = model.predict(pad_seq, verbose=0)
    predicted_index  = np.argmax(prediction, axis=1)[0]
    confidence_score = round(float(np.max(prediction)) * 100, 2)
    predicted_label  = le.inverse_transform([predicted_index])[0]

    return {
        "Prediction"      : predicted_label,
        "Confidence_Score": confidence_score
    }