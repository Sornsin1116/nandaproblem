import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ==============================
# Paths
# ==============================
DATA_PATH = "data/processed"
X_train_file = os.path.join(DATA_PATH, "X_train.csv")
X_test_file = os.path.join(DATA_PATH, "X_test.csv")
y_train_file = os.path.join(DATA_PATH, "y_train.csv")
y_test_file = os.path.join(DATA_PATH, "y_test.csv")
mapping_file = os.path.join(DATA_PATH, "disease_mapping.json")

# ==============================
# Load processed data
# ==============================
X_train = pd.read_csv(X_train_file)
X_test = pd.read_csv(X_test_file)
y_train = pd.read_csv(y_train_file).squeeze()
y_test = pd.read_csv(y_test_file).squeeze()

with open(mapping_file, "r") as f:
    label_mapping = json.load(f)

inverse_mapping = {v: k for k, v in label_mapping.items()}

print("✅ Loaded processed data successfully.")
print(f"📊 X_train shape: {X_train.shape}")
print(f"🎯 y_train shape: {y_train.shape}")
print(f"🩺 Total diseases: {len(label_mapping)}\n")

# ==============================
# Train model
# ==============================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ==============================
# Predict function for multiple patients
# ==============================
def predict_patients(patients: list[dict[str, int]]):
    df_patients = pd.DataFrame(patients)
    preds = model.predict(df_patients)
    preds_prob = model.predict_proba(df_patients)

    results = []
    for i, patient in enumerate(patients):
        disease_id = preds[i]
        disease_name = inverse_mapping[disease_id]
        prob_dict = {inverse_mapping[j]: float(preds_prob[i][j]) for j in range(len(preds_prob[i]))}

        results.append({
            "patient_symptoms": patient,
            "predicted_disease": disease_name,
            "disease_probabilities": prob_dict
        })
    return results

# ==============================
# Example usage
# ==============================
example_patients = [
    {'Symptom_1': 1, 'Symptom_2': 0, 'Symptom_3': 1, 'Symptom_4': 0, 'Symptom_5': 1,
     'Symptom_6': 0, 'Symptom_7': 0, 'Symptom_8': 0, 'Symptom_9': 0, 'Symptom_10': 0,
     'Symptom_11': 0, 'Symptom_12': 0, 'Symptom_13': 0, 'Symptom_14': 0, 'Symptom_15': 0,
     'Symptom_16': 0, 'Symptom_17': 0},
    {'Symptom_1': 0, 'Symptom_2': 1, 'Symptom_3': 0, 'Symptom_4': 1, 'Symptom_5': 0,
     'Symptom_6': 1, 'Symptom_7': 0, 'Symptom_8': 1, 'Symptom_9': 0, 'Symptom_10': 0,
     'Symptom_11': 0, 'Symptom_12': 0, 'Symptom_13': 0, 'Symptom_14': 0, 'Symptom_15': 0,
     'Symptom_16': 0, 'Symptom_17': 0}
]

results = predict_patients(example_patients)

# ==============================
# Pretty Output
# ==============================
for idx, r in enumerate(results, 1):
    print("═" * 60)
    print(f"👩‍⚕️ Patient #{idx}")
    print("🧩 Symptoms:")
    for k, v in r["patient_symptoms"].items():
        if v == 1:
            print(f"   ✅ {k}")
    print(f"\n🔮 Predicted Disease: 🌡️  {r['predicted_disease']}")
    print("\n📈 Top 5 Possible Diseases:")
    top5 = sorted(r["disease_probabilities"].items(), key=lambda x: x[1], reverse=True)[:5]
    for disease, prob in top5:
        print(f"   • {disease:<35} →  {prob:.3f}")
    print("═" * 60 + "\n")
