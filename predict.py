import joblib
import pandas as pd

model = joblib.load("career_model.pkl")
target_encoder = joblib.load("target_encoder.pkl")
encoder_bundle = joblib.load("feature_encoders.pkl")

label_encoders = encoder_bundle["label_encoders"]
project_binarizer = encoder_bundle["project_binarizer"]
base_feature_columns = encoder_bundle["base_feature_columns"]
project_prefix = encoder_bundle.get("project_prefix", "project__")


def split_projects(value: str) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def encode_user_data(user_data: dict) -> pd.DataFrame:
    df = pd.DataFrame([user_data])

    for col, encoder in label_encoders.items():
        value = str(df.at[0, col])
        if value not in encoder.classes_:
            print(f"Warning: '{value}' not seen in training for {col}. Using fallback '{encoder.classes_[0]}'.")
            value = encoder.classes_[0]
        df[col] = encoder.transform([value])

    input_projects = split_projects(df.at[0, "Projects"])
    safe_projects = [p for p in input_projects if p in project_binarizer.classes_]
    if not safe_projects:
        safe_projects = [project_binarizer.classes_[0]]

    project_matrix = project_binarizer.transform([safe_projects])
    project_columns = [f"{project_prefix}{name}" for name in project_binarizer.classes_]
    projects_df = pd.DataFrame(project_matrix, columns=project_columns)

    encoded = pd.concat([df[base_feature_columns], projects_df], axis=1)

    if hasattr(model, "feature_names_in_"):
        encoded = encoded.reindex(columns=model.feature_names_in_, fill_value=0)

    return encoded

user_data = {
    "Age": 21,
    "GPA": 3.6,
    "Interested Domain": "Web Development",
    "Projects": "Full-Stack Web App; Front-End Development; Network Security",
    "Python": "Weak",
    "SQL": "Strong",
    "Java": "Weak"
}

encoded_data = encode_user_data(user_data)
prediction = model.predict(encoded_data)
predicted_label = target_encoder.inverse_transform(prediction)

for k, v in user_data.items():
    print(f"{k}: {v}")
print("\nPredicted Career:", predicted_label[0])