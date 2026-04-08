import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, label_binarize
from sklearn.tree import DecisionTreeClassifier


BASE_FEATURE_COLUMNS = ["Age", "GPA", "Interested Domain", "Python", "SQL", "Java"]
CAT_FEATURE_COLUMNS = ["Interested Domain", "Python", "SQL", "Java"]
PROJECT_COLUMN = "Projects"
PROJECT_PREFIX = "project__"
SHOW_PLOTS = True
SAVE_INTERACTIVE_IMPORTANCE_HTML = True


def split_projects(value: str) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def build_domain_core_projects(df: pd.DataFrame) -> dict[str, set[str]]:
    core = {}
    for domain, group in df.groupby("Interested Domain"):
        domain_projects = set()
        for value in group[PROJECT_COLUMN]:
            projects = split_projects(value)
            if projects:
                # The first item is the original domain-aligned project in the synthesized dataset.
                domain_projects.add(projects[0])
        core[domain] = domain_projects
    return core


def build_sample_weights(df: pd.DataFrame, domain_core_projects: dict[str, set[str]]) -> pd.Series:
    weights = []
    for _, row in df.iterrows():
        projects = split_projects(row[PROJECT_COLUMN])
        if not projects:
            weights.append(1.0)
            continue

        domain = row["Interested Domain"]
        core = domain_core_projects.get(domain, set())
        match_count = sum(1 for p in projects if p in core)
        match_ratio = match_count / len(projects)
        extra_count = max(0, len(projects) - 1)

        # Slightly prioritize rows where project signals align with the chosen domain.
        weight = 1.0 + (0.35 * match_ratio) + (0.05 * extra_count)
        weights.append(weight)

    return pd.Series(weights, index=df.index)


def encode_features(
    df: pd.DataFrame,
    label_encoders: dict[str, LabelEncoder] | None = None,
    project_binarizer: MultiLabelBinarizer | None = None,
    fit: bool = False,
):
    work = df.copy()

    if label_encoders is None:
        label_encoders = {}

    for col in CAT_FEATURE_COLUMNS:
        if fit:
            encoder = LabelEncoder()
            work[col] = encoder.fit_transform(work[col].astype(str))
            label_encoders[col] = encoder
        else:
            encoder = label_encoders[col]
            work[col] = work[col].astype(str).apply(lambda value: value if value in encoder.classes_ else encoder.classes_[0])
            work[col] = encoder.transform(work[col])

    project_lists = work[PROJECT_COLUMN].apply(split_projects).tolist()
    if fit:
        project_binarizer = MultiLabelBinarizer()
        project_matrix = project_binarizer.fit_transform(project_lists)
    else:
        assert project_binarizer is not None
        safe_project_lists = [[p for p in projects if p in project_binarizer.classes_] for projects in project_lists]
        project_matrix = project_binarizer.transform(safe_project_lists)

    project_cols = [f"{PROJECT_PREFIX}{name}" for name in project_binarizer.classes_]
    project_df = pd.DataFrame(project_matrix, columns=project_cols, index=work.index)

    X = pd.concat([work[BASE_FEATURE_COLUMNS], project_df], axis=1)
    return X, label_encoders, project_binarizer


def evaluate_model(y_true, y_pred, model_name):
    print(f"\n=== {model_name} Performance ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average="weighted", zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, average="weighted", zero_division=0))
    print("F1 Score:", f1_score(y_true, y_pred, average="weighted", zero_division=0))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, zero_division=0))


def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def evaluate_roc_auc(model, X_eval, y_eval, model_name):
    if not hasattr(model, "predict_proba"):
        print(f"\n{model_name} does not support predict_proba. Skipping ROC/AUC.")
        return

    model_classes = model.classes_
    known_mask = np.isin(y_eval, model_classes)
    if not np.all(known_mask):
        print(f"\nWarning: {model_name} ROC/AUC skipped for unseen test classes not present in training.")

    X_eval_known = X_eval[known_mask]
    y_eval_known = y_eval[known_mask]
    if len(y_eval_known) == 0:
        print(f"\n{model_name} ROC/AUC cannot be computed (no supported classes in test set).")
        return

    y_scores = model.predict_proba(X_eval_known)
    y_bin = label_binarize(y_eval_known, classes=model_classes)

    per_class_aucs = {}
    per_class_fpr_tpr = {}

    for class_index, class_value in enumerate(model_classes):
        class_targets = y_bin[:, class_index]
        positives = class_targets.sum()
        negatives = len(class_targets) - positives

        # ROC needs both positive and negative samples for this one-vs-rest view.
        if positives == 0 or negatives == 0:
            continue

        fpr, tpr, _ = roc_curve(class_targets, y_scores[:, class_index])
        class_auc = auc(fpr, tpr)
        per_class_aucs[int(class_value)] = class_auc
        per_class_fpr_tpr[int(class_value)] = (fpr, tpr)

    if not per_class_aucs:
        print(f"\n{model_name} ROC/AUC cannot be computed (insufficient class distribution in test set).")
        return

    class_support = pd.Series(y_eval_known).value_counts()
    valid_classes = list(per_class_aucs.keys())
    macro_auc = float(np.mean([per_class_aucs[c] for c in valid_classes]))
    weighted_auc = float(
        np.average(
            [per_class_aucs[c] for c in valid_classes],
            weights=[class_support.get(c, 0) for c in valid_classes],
        )
    )

    print(f"\n{model_name} ROC/AUC Summary:")
    print("AUC (Macro One-vs-Rest):", round(macro_auc, 4))
    print("AUC (Weighted One-vs-Rest):", round(weighted_auc, 4))

    for class_value in valid_classes:
        print(f"Class {class_value} AUC:", round(per_class_aucs[class_value], 4))

    if SHOW_PLOTS:
        plt.figure(figsize=(8, 6))
        for class_value in valid_classes:
            fpr, tpr = per_class_fpr_tpr[class_value]
            plt.plot(fpr, tpr, label=f"Class {class_value} (AUC={per_class_aucs[class_value]:.2f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.title(f"ROC Curve - {model_name} (One-vs-Rest)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.show()
    else:
        plt.close("all")


def plot_grouped_feature_importance(importance: pd.Series, output_html_path: str):
    project_mask = importance.index.str.startswith(PROJECT_PREFIX)
    project_importance = importance[project_mask].sort_values(ascending=False)
    project_total = float(project_importance.sum())

    grouped_values = {
        "Age": float(importance.get("Age", 0.0)),
        "GPA": float(importance.get("GPA", 0.0)),
        "Interested Domain": float(importance.get("Interested Domain", 0.0)),
        "Python": float(importance.get("Python", 0.0)),
        "SQL": float(importance.get("SQL", 0.0)),
        "Java": float(importance.get("Java", 0.0)),
        "Projects": project_total,
    }

    project_hover_lines = []
    preview_count = 12
    for idx, (name, value) in enumerate(project_importance.items(), start=1):
        if idx > preview_count:
            break
        clean_name = name.replace(PROJECT_PREFIX, "", 1)
        project_hover_lines.append(f"{idx}. {clean_name}: {value:.4f}")

    if len(project_importance) > preview_count:
        project_hover_lines.append(f"... +{len(project_importance) - preview_count} more projects")

    project_hover_text = "<br>".join(project_hover_lines) if project_hover_lines else "No project features"

    x_labels = list(grouped_values.keys())
    y_values = [grouped_values[label] for label in x_labels]
    hover_text = []
    for label in x_labels:
        if label == "Projects":
            hover_text.append(f"Projects Total: {grouped_values[label]:.4f}<br><br>{project_hover_text}")
        else:
            hover_text.append(f"{label}: {grouped_values[label]:.4f}")

    fig = go.Figure(
        data=[
            go.Bar(
                x=x_labels,
                y=y_values,
                marker_color=["#2E8B57" if label != "Projects" else "#1E3D59" for label in x_labels],
                hovertext=hover_text,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Grouped Feature Importance (Projects Aggregated)",
        xaxis_title="Feature Group",
        yaxis_title="Importance",
        template="plotly_white",
        hoverlabel={"align": "left"},
    )

    if SAVE_INTERACTIVE_IMPORTANCE_HTML:
        fig.write_html(output_html_path, include_plotlyjs="cdn")
        print(f"\nSaved interactive grouped feature-importance chart: {output_html_path}")

    if SHOW_PLOTS:
        fig.show()


def main():
    df = pd.read_csv("cs_students.csv")
    df = df.drop_duplicates()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[df["Age"] <= 30].copy()

    drop_cols = ["Student ID", "Name", "Gender", "Major"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    df.to_csv("cleaned_dataset.csv", index=False)

    y_raw = df["Future Career"].copy()
    domain_core_projects = build_domain_core_projects(df)
    sample_weights = build_sample_weights(df, domain_core_projects)

    X, label_encoders, project_binarizer = encode_features(df, fit=True)
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_raw)

    class_counts = pd.Series(y).value_counts()
    stratify_target = y if class_counts.min() >= 2 else None
    if stratify_target is None:
        print("Warning: Some classes have only one sample. Using non-stratified split.")

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X,
        y,
        sample_weights,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target,
    )

    dt_model = DecisionTreeClassifier(random_state=42)
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)

    dt_model.fit(X_train, y_train, sample_weight=w_train)
    rf_model.fit(X_train, y_train, sample_weight=w_train)

    dt_pred = dt_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    evaluate_model(y_test, dt_pred, "Decision Tree")
    evaluate_model(y_test, rf_pred, "Random Forest")
    evaluate_roc_auc(dt_model, X_test, y_test, "Decision Tree")
    evaluate_roc_auc(rf_model, X_test, y_test, "Random Forest")

    plot_confusion(y_test, dt_pred, "Decision Tree Confusion Matrix")
    plot_confusion(y_test, rf_pred, "Random Forest Confusion Matrix")

    importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 20 Feature Importances:\n")
    print(importance.head(20))

    plot_grouped_feature_importance(importance, output_html_path="feature_importance_grouped.html")

    encoder_bundle = {
        "label_encoders": label_encoders,
        "project_binarizer": project_binarizer,
        "base_feature_columns": BASE_FEATURE_COLUMNS,
        "project_prefix": PROJECT_PREFIX,
        "project_weighting": {
            "match_ratio_boost": 0.35,
            "extra_project_boost": 0.05,
        },
    }

    joblib.dump(rf_model, "career_model.pkl")
    joblib.dump(target_encoder, "target_encoder.pkl")
    joblib.dump(encoder_bundle, "feature_encoders.pkl")
    print("\nSaved updated artifacts: career_model.pkl, target_encoder.pkl, feature_encoders.pkl")


if __name__ == "__main__":
    main()