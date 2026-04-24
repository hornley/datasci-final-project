import os
import json
import time
import joblib
import hashlib
import secrets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, precision_score, recall_score, f1_score, confusion_matrix, roc_curve


st.set_page_config(page_title="Career Prediction Dashboard", page_icon="📊", layout="wide")


DATASET_PATH = "cs_students.csv"
MODEL_PATH = "career_model.pkl"
TARGET_ENCODER_PATH = "target_encoder.pkl"
FEATURE_ENCODERS_PATH = "feature_encoders.pkl"
USERS_DB_PATH = "users.json"

BASE_FEATURE_COLUMNS = ["Age", "GPA", "Interested Domain", "Python", "SQL", "Java"]
CAT_FEATURE_COLUMNS = ["Interested Domain", "Python", "SQL", "Java"]
PROJECT_COLUMN = "Projects"
PROJECT_PREFIX = "project__"


def init_auth_state() -> None:
    if "is_authenticated" not in st.session_state:
        st.session_state["is_authenticated"] = False
    if "current_user" not in st.session_state:
        st.session_state["current_user"] = None


def load_users() -> dict:
    if not os.path.exists(USERS_DB_PATH):
        return {}

    try:
        with open(USERS_DB_PATH, "r", encoding="utf-8") as f:
            users = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    return users if isinstance(users, dict) else {}


def save_users(users: dict) -> None:
    with open(USERS_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def hash_password(password: str, salt: str | None = None) -> str:
    safe_salt = salt or secrets.token_hex(16)
    digest = hashlib.sha256(f"{safe_salt}:{password}".encode("utf-8")).hexdigest()
    return f"{safe_salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    parts = stored_hash.split("$", 1)
    if len(parts) != 2:
        return False

    salt, _ = parts
    return hash_password(password, salt) == stored_hash


def selectbox_index(options: list[str], value: str | None) -> int | None:
    if value is None:
        return None
    return options.index(value) if value in options else None


def normalize_profile_projects(raw_projects) -> list[str]:
    if isinstance(raw_projects, list):
        return [str(p).strip() for p in raw_projects if str(p).strip()]
    if isinstance(raw_projects, str):
        return split_projects(raw_projects)
    return []


def build_profile_payload(
    age: int,
    gpa: float,
    interested_domain: str,
    projects: list[str],
    python_skill: str,
    sql_skill: str,
    java_skill: str,
) -> dict:
    return {
        "Age": int(age),
        "GPA": float(gpa),
        "Interested Domain": interested_domain,
        "Projects": projects,
        "Python": python_skill,
        "SQL": sql_skill,
        "Java": java_skill,
    }


def render_auth_ui(saved_artifacts: dict | None) -> bool:
    left_spacer, center_container, right_spacer = st.columns([1, 1.8, 1])
    with center_container:
        auth_wrapper = st.container(border=True)

    with auth_wrapper:
        st.markdown(
            """
            <h1 style='text-align: center; margin-bottom: 0.2rem;'>Career Model Dashboard</h1>
            <p style='text-align: center; color: #666;'>
                View training results and predict your future career based on your profile.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.subheader("Account Security")
        st.caption("Create an account and log in. You can optionally save your predictor profile during signup.")

        auth_login_tab, auth_signup_tab = st.tabs(["Login", "Create Account"])

        with auth_login_tab:
            with st.form("login_form"):
                login_username = st.text_input("Username", key="login_username").strip()
                login_password = st.text_input("Password", type="password", key="login_password")
                login_submit = st.form_submit_button("Login", use_container_width=True)

                if login_submit:
                    users = load_users()
                    user_record = users.get(login_username)
                    if user_record is None:
                        st.error("Username not found.")
                    elif not verify_password(login_password, user_record.get("password_hash", "")):
                        st.error("Invalid password.")
                    else:
                        st.session_state["is_authenticated"] = True
                        st.session_state["current_user"] = login_username
                        st.success("Login successful.")
                        st.rerun()

        with auth_signup_tab:
            profile_ready = (
                saved_artifacts is not None
                and "feature_encoders" in saved_artifacts
                and "label_encoders" in saved_artifacts["feature_encoders"]
                and "project_binarizer" in saved_artifacts["feature_encoders"]
            )
            label_encoders = saved_artifacts["feature_encoders"].get("label_encoders", {}) if profile_ready else {}
            project_binarizer = saved_artifacts["feature_encoders"].get("project_binarizer") if profile_ready else None
            create_profile_now = st.checkbox(
                "Create predictor profile now",
                value=True,
                key="signup_create_profile_now",
            )

            with st.form("signup_form"):
                signup_username = st.text_input("Username", key="signup_username").strip()
                signup_password = st.text_input("Password", type="password", key="signup_password")
                signup_confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")

                profile_payload = None
                if create_profile_now:
                    if not profile_ready:
                        st.info("Model artifacts are not available yet. You can create an account now and save a profile later in Career Predictor.")
                    else:
                        st.markdown("### Profile Information")
                        left_profile, right_profile = st.columns(2)
                        with left_profile:
                            signup_age = st.slider("Age", min_value=16, max_value=30, value=21, key="signup_age")
                            signup_gpa = st.slider("GPA", min_value=1.0, max_value=4.0, value=3.5, step=0.1, key="signup_gpa")
                            signup_domain = st.selectbox(
                                "Interested Domain",
                                list(label_encoders["Interested Domain"].classes_),
                                index=None,
                                placeholder="Type or select a domain",
                                key="signup_domain",
                            )
                        with right_profile:
                            signup_projects = st.multiselect(
                                "Projects (select one or more)",
                                list(project_binarizer.classes_),
                                key="signup_projects",
                            )
                            signup_python = st.selectbox(
                                "Python",
                                list(label_encoders["Python"].classes_),
                                index=None,
                                placeholder="Select Python skill",
                                key="signup_python",
                            )
                            signup_sql = st.selectbox(
                                "SQL",
                                list(label_encoders["SQL"].classes_),
                                index=None,
                                placeholder="Select SQL skill",
                                key="signup_sql",
                            )
                            signup_java = st.selectbox(
                                "Java",
                                list(label_encoders["Java"].classes_),
                                index=None,
                                placeholder="Select Java skill",
                                key="signup_java",
                            )

                        profile_missing = []
                        if signup_domain is None:
                            profile_missing.append("Interested Domain")
                        if not signup_projects:
                            profile_missing.append("Projects")
                        if signup_python is None:
                            profile_missing.append("Python")
                        if signup_sql is None:
                            profile_missing.append("SQL")
                        if signup_java is None:
                            profile_missing.append("Java")

                        if not profile_missing:
                            profile_payload = build_profile_payload(
                                age=signup_age,
                                gpa=signup_gpa,
                                interested_domain=signup_domain,
                                projects=signup_projects,
                                python_skill=signup_python,
                                sql_skill=signup_sql,
                                java_skill=signup_java,
                            )

                signup_submit = st.form_submit_button("Create Account", use_container_width=True)

                if signup_submit:
                    users = load_users()

                    if len(signup_username) < 3:
                        st.error("Username must be at least 3 characters.")
                        return False
                    if signup_username in users:
                        st.error("Username already exists.")
                        return False
                    if len(signup_password) < 6:
                        st.error("Password must be at least 6 characters.")
                        return False
                    if signup_password != signup_confirm_password:
                        st.error("Passwords do not match.")
                        return False

                    if create_profile_now and profile_ready and profile_payload is None:
                        st.error("Please complete all profile fields before creating your account.")
                        return False

                    users[signup_username] = {
                        "password_hash": hash_password(signup_password),
                        "profile": profile_payload,
                    }
                    save_users(users)

                    st.session_state["is_authenticated"] = True
                    st.session_state["current_user"] = signup_username
                    st.success("Account created successfully.")
                    st.rerun()

    return st.session_state.get("is_authenticated", False)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df = df.drop_duplicates()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[df["Age"] <= 30].copy()

    drop_cols = ["Student ID", "Name", "Gender", "Major"]
    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    if existing_drop_cols:
        df = df.drop(columns=existing_drop_cols)

    return df


@st.cache_data
def train_and_evaluate(df: pd.DataFrame):
    work = df.copy()
    y = work["Future Career"].copy()

    domain_core_projects = build_domain_core_projects(work)
    sample_weights = build_sample_weights(work, domain_core_projects)

    X, feature_encoders, project_binarizer = encode_features(work, fit=True)

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    class_counts = pd.Series(y_encoded).value_counts()
    stratify_target = y_encoded if class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y_encoded, sample_weights, test_size=0.2, random_state=42, stratify=stratify_target
    )

    dt_model = DecisionTreeClassifier(random_state=42)
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)

    dt_model.fit(X_train, y_train, sample_weight=w_train)
    rf_model.fit(X_train, y_train, sample_weight=w_train)

    dt_pred = dt_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    dt_roc = compute_roc_auc_data(dt_model, X_test, y_test)
    rf_roc = compute_roc_auc_data(rf_model, X_test, y_test)

    metrics = {
        "Decision Tree": {
            "accuracy": accuracy_score(y_test, dt_pred),
            "precision": precision_score(y_test, dt_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, dt_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, dt_pred, average="weighted", zero_division=0),
            "cm": confusion_matrix(y_test, dt_pred),
            "roc": dt_roc,
        },
        "Random Forest": {
            "accuracy": accuracy_score(y_test, rf_pred),
            "precision": precision_score(y_test, rf_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, rf_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, rf_pred, average="weighted", zero_division=0),
            "cm": confusion_matrix(y_test, rf_pred),
            "roc": rf_roc,
        },
    }

    importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    artifacts = {
        "rf_model": rf_model,
        "target_encoder": target_encoder,
        "feature_encoders": {
            "label_encoders": feature_encoders,
            "project_binarizer": project_binarizer,
            "base_feature_columns": BASE_FEATURE_COLUMNS,
            "project_prefix": PROJECT_PREFIX,
            "project_weighting": {
                "match_ratio_boost": 0.35,
                "extra_project_boost": 0.05,
            },
        },
        "class_labels": target_encoder.classes_,
    }

    return metrics, importance, artifacts


def compute_roc_auc_data(model, X_eval: pd.DataFrame, y_eval: np.ndarray) -> dict:
    if not hasattr(model, "predict_proba"):
        return {
            "macro_auc": None,
            "weighted_auc": None,
            "curves": {},
            "warning": "Model does not support probability prediction.",
        }

    model_classes = model.classes_
    known_mask = np.isin(y_eval, model_classes)
    warning = None
    if not np.all(known_mask):
        warning = "Some unseen test classes were excluded from ROC/AUC computation."

    X_known = X_eval[known_mask]
    y_known = y_eval[known_mask]
    if len(y_known) == 0:
        return {
            "macro_auc": None,
            "weighted_auc": None,
            "curves": {},
            "warning": "No supported classes available for ROC/AUC.",
        }

    y_scores = model.predict_proba(X_known)
    y_bin = label_binarize(y_known, classes=model_classes)

    per_class_auc = {}
    curves = {}
    for idx, class_value in enumerate(model_classes):
        targets = y_bin[:, idx]
        positives = targets.sum()
        negatives = len(targets) - positives
        if positives == 0 or negatives == 0:
            continue

        fpr, tpr, _ = roc_curve(targets, y_scores[:, idx])
        class_auc = float(auc(fpr, tpr))
        class_key = str(class_value)
        per_class_auc[class_key] = class_auc
        curves[class_key] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": class_auc,
        }

    if not per_class_auc:
        return {
            "macro_auc": None,
            "weighted_auc": None,
            "curves": {},
            "warning": "Insufficient class distribution for ROC/AUC.",
        }

    support = pd.Series(y_known).value_counts()
    valid_class_ids = [int(key) for key in per_class_auc.keys()]
    macro_auc = float(np.mean(list(per_class_auc.values())))
    weighted_auc = float(
        np.average(
            list(per_class_auc.values()),
            weights=[support.get(class_id, 0) for class_id in valid_class_ids],
        )
    )

    return {
        "macro_auc": macro_auc,
        "weighted_auc": weighted_auc,
        "curves": curves,
        "warning": warning,
    }


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
        weights.append(1.0 + (0.35 * match_ratio) + (0.05 * extra_count))

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
        safe_lists = [[p for p in projects if p in project_binarizer.classes_] for projects in project_lists]
        project_matrix = project_binarizer.transform(safe_lists)

    project_cols = [f"{PROJECT_PREFIX}{name}" for name in project_binarizer.classes_]
    project_df = pd.DataFrame(project_matrix, columns=project_cols, index=work.index)

    X = pd.concat([work[BASE_FEATURE_COLUMNS], project_df], axis=1)
    return X, label_encoders, project_binarizer


def save_artifacts(artifacts) -> None:
    joblib.dump(artifacts["rf_model"], MODEL_PATH)
    joblib.dump(artifacts["target_encoder"], TARGET_ENCODER_PATH)
    joblib.dump(artifacts["feature_encoders"], FEATURE_ENCODERS_PATH)


def load_saved_artifacts():
    if not (
        os.path.exists(MODEL_PATH)
        and os.path.exists(TARGET_ENCODER_PATH)
        and os.path.exists(FEATURE_ENCODERS_PATH)
    ):
        return None

    saved = {
        "model": joblib.load(MODEL_PATH),
        "target_encoder": joblib.load(TARGET_ENCODER_PATH),
        "feature_encoders": joblib.load(FEATURE_ENCODERS_PATH),
    }

    return saved


def plot_confusion_matrix(cm: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    left_spacer, center_plot, right_spacer = st.columns([1, 2, 1])
    with center_plot:
        st.pyplot(fig, use_container_width=False)


def plot_roc_curve(roc_data: dict, title: str) -> None:
    curves = roc_data.get("curves", {})
    if not curves:
        st.info("ROC curve is not available for this split/model.")
        return

    fig = go.Figure()
    for class_name, class_data in curves.items():
        fig.add_trace(
            go.Scatter(
                x=class_data["fpr"],
                y=class_data["tpr"],
                mode="lines",
                name=f"Class {class_name} (AUC={class_data['auc']:.2f})",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line={"dash": "dash", "color": "gray"},
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        legend={"font": {"size": 10}},
        width=760,
        height=420,
    )

    left_spacer, center_plot, right_spacer = st.columns([1, 2, 1])
    with center_plot:
        st.plotly_chart(fig, use_container_width=False)


def build_grouped_feature_importance_figure(importance: pd.Series) -> tuple[go.Figure, pd.DataFrame]:
    project_mask = importance.index.str.startswith(PROJECT_PREFIX)
    project_importance = importance[project_mask].sort_values(ascending=False)
    project_total = float(project_importance.sum())

    project_breakdown_df = pd.DataFrame(
        {
            "Project": [name.replace(PROJECT_PREFIX, "", 1) for name in project_importance.index],
            "Importance": [float(v) for v in project_importance.values],
        }
    )

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
        project_name = name.replace(PROJECT_PREFIX, "", 1)
        project_hover_lines.append(f"{idx}. {project_name}: {value:.4f}")

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
        width=760,
        height=420,
        hoverlabel={"align": "left"},
    )
    return fig, project_breakdown_df


def encode_user_input(input_df: pd.DataFrame, feature_encoders: dict, model) -> pd.DataFrame:
    label_encoders = feature_encoders["label_encoders"]
    project_binarizer = feature_encoders["project_binarizer"]
    base_feature_columns = feature_encoders["base_feature_columns"]
    project_prefix = feature_encoders.get("project_prefix", PROJECT_PREFIX)

    encoded_df = input_df.copy()
    for col, encoder in label_encoders.items():
        value = str(encoded_df.at[0, col])
        if value not in encoder.classes_:
            value = encoder.classes_[0]
        encoded_df[col] = encoder.transform([value])

    selected_projects = encoded_df.at[0, PROJECT_COLUMN]
    project_list = selected_projects if isinstance(selected_projects, list) else split_projects(str(selected_projects))
    safe_projects = [p for p in project_list if p in project_binarizer.classes_]
    if not safe_projects:
        safe_projects = [project_binarizer.classes_[0]]

    project_matrix = project_binarizer.transform([safe_projects])
    project_columns = [f"{project_prefix}{name}" for name in project_binarizer.classes_]
    project_df = pd.DataFrame(project_matrix, columns=project_columns)

    encoded = pd.concat([encoded_df[base_feature_columns], project_df], axis=1)
    if hasattr(model, "feature_names_in_"):
        encoded = encoded.reindex(columns=model.feature_names_in_, fill_value=0)

    return encoded


@st.dialog("Prediction Result")
def show_prediction_dialog() -> None:
    payload = st.session_state.get("prediction_payload")
    if not payload:
        st.info("No prediction available.")
        return

    st.success(f"Predicted Career: {payload['predicted_career']}")
    if payload.get("prob_df") is not None:
        st.markdown("### Top Career Matches")
        st.dataframe(payload["prob_df"], use_container_width=True)

    if st.button("Close", use_container_width=True):
        st.session_state["show_prediction_dialog"] = False
        st.session_state.pop("prediction_payload", None)
        st.rerun()


def main() -> None:
    init_auth_state()

    try:
        df = load_dataset()
    except FileNotFoundError:
        st.error("Dataset file not found. Ensure cs_students.csv exists in the project folder.")
        return

    saved_for_auth = load_saved_artifacts()
    if not st.session_state["is_authenticated"]:
        render_auth_ui(saved_for_auth)
        return

    st.title("Career Model Dashboard")
    st.caption("View training results and predict your future career based on your profile.")

    st.sidebar.success(f"Logged in as: {st.session_state['current_user']}")
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state["is_authenticated"] = False
        st.session_state["current_user"] = None
        st.rerun()

    tab1, tab2 = st.tabs(["Training Results", "Career Predictor"])

    if "show_prediction_dialog" not in st.session_state:
        st.session_state["show_prediction_dialog"] = False

    with tab1:
        st.subheader("Model Training Results")
        metrics, importance, artifacts = train_and_evaluate(df)

        selected_model = st.selectbox("Choose model", ["Decision Tree", "Random Forest"])
        model_metrics = metrics[selected_model]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{model_metrics['accuracy']:.3f}")
        c2.metric("Precision", f"{model_metrics['precision']:.3f}")
        c3.metric("Recall", f"{model_metrics['recall']:.3f}")
        c4.metric("F1 Score", f"{model_metrics['f1']:.3f}")

        roc_metrics = model_metrics["roc"]
        auc_macro = roc_metrics.get("macro_auc")
        auc_weighted = roc_metrics.get("weighted_auc")
        c5, c6 = st.columns(2)
        c5.metric("AUC (Macro OVR)", f"{auc_macro:.3f}" if auc_macro is not None else "N/A")
        c6.metric("AUC (Weighted OVR)", f"{auc_weighted:.3f}" if auc_weighted is not None else "N/A")

        if roc_metrics.get("warning"):
            st.caption(roc_metrics["warning"])

        st.markdown("### Confusion Matrix")
        plot_confusion_matrix(model_metrics["cm"], f"{selected_model} Confusion Matrix")

        st.markdown("### ROC Curve")
        plot_roc_curve(roc_metrics, f"{selected_model} ROC Curve (One-vs-Rest)")

        st.markdown("### Grouped Feature Importance")
        grouped_fig, project_breakdown_df = build_grouped_feature_importance_figure(importance)
        left_spacer, center_plot, right_spacer = st.columns([1, 2, 1])
        with center_plot:
            st.plotly_chart(grouped_fig, use_container_width=False)

        with st.expander("View Full Project Importance Breakdown"):
            st.dataframe(project_breakdown_df, use_container_width=True)

        if st.button("Train and Save Random Forest Model", use_container_width=True):
            save_artifacts(artifacts)
            st.success("Model and encoders saved: career_model.pkl, target_encoder.pkl, feature_encoders.pkl")

    with tab2:
        st.subheader("Predict Your Career")

        saved = saved_for_auth
        if saved is None:
            st.warning("Saved model files not found. Use the 'Train and Save Random Forest Model' button in Training Results first.")
            return

        model = saved["model"]
        target_encoder = saved["target_encoder"]
        feature_encoders = saved["feature_encoders"]

        if "label_encoders" not in feature_encoders or "project_binarizer" not in feature_encoders:
            st.error("Saved encoders are in an old format. Run training.py or click 'Train and Save Random Forest Model' to refresh artifacts.")
            return

        label_encoders = feature_encoders["label_encoders"]
        project_binarizer = feature_encoders["project_binarizer"]

        users = load_users()
        current_user = st.session_state.get("current_user")
        user_record = users.get(current_user, {})
        user_profile = user_record.get("profile") or {}
        profile_projects = normalize_profile_projects(user_profile.get("Projects"))

        if user_profile:
            st.caption("Your saved profile was loaded. You can update values and save again anytime.")
        else:
            st.caption("No saved profile found. Fill the form and click 'Save Inputs to My Profile' for faster future predictions.")

        domain_options = list(label_encoders["Interested Domain"].classes_)
        python_options = list(label_encoders["Python"].classes_)
        sql_options = list(label_encoders["SQL"].classes_)
        java_options = list(label_encoders["Java"].classes_)
        project_options = list(project_binarizer.classes_)

        default_age = max(16, min(30, int(user_profile.get("Age", 21))))
        default_gpa = max(1.0, min(4.0, float(user_profile.get("GPA", 3.5))))
        default_projects = [p for p in profile_projects if p in project_options]

        left, right = st.columns(2)
        with left:
            age = st.slider("Age", min_value=16, max_value=30, value=default_age)
            gpa = st.slider("GPA", min_value=1.0, max_value=4.0, value=default_gpa, step=0.1)
            interested_domain = st.selectbox(
                "Interested Domain",
                domain_options,
                index=selectbox_index(domain_options, user_profile.get("Interested Domain")),
                placeholder="Type or select a domain",
            )

        with right:
            projects = st.multiselect(
                "Projects (select one or more)",
                project_options,
                default=default_projects,
            )
            python_skill = st.selectbox(
                "Python",
                python_options,
                index=selectbox_index(python_options, user_profile.get("Python")),
                placeholder="Select Python skill",
            )
            sql_skill = st.selectbox(
                "SQL",
                sql_options,
                index=selectbox_index(sql_options, user_profile.get("SQL")),
                placeholder="Select SQL skill",
            )
            java_skill = st.selectbox(
                "Java",
                java_options,
                index=selectbox_index(java_options, user_profile.get("Java")),
                placeholder="Select Java skill",
            )

        if st.button("Save Inputs to My Profile", use_container_width=True):
            missing_profile_fields = []
            if interested_domain is None:
                missing_profile_fields.append("Interested Domain")
            if not projects:
                missing_profile_fields.append("Projects")
            if python_skill is None:
                missing_profile_fields.append("Python")
            if sql_skill is None:
                missing_profile_fields.append("SQL")
            if java_skill is None:
                missing_profile_fields.append("Java")

            if missing_profile_fields:
                st.warning(f"Please select: {', '.join(missing_profile_fields)}")
            else:
                if not current_user or current_user not in users:
                    st.error("Your account session is out of sync. Please log out and log in again.")
                    return

                users[current_user]["profile"] = build_profile_payload(
                    age=age,
                    gpa=gpa,
                    interested_domain=interested_domain,
                    projects=projects,
                    python_skill=python_skill,
                    sql_skill=sql_skill,
                    java_skill=java_skill,
                )
                save_users(users)
                st.success("Profile saved. Your inputs will be pre-filled the next time you log in.")

        if st.button("Predict Career", type="primary", use_container_width=True):
            missing_inputs = []
            if interested_domain is None:
                missing_inputs.append("Interested Domain")
            if not projects:
                missing_inputs.append("Projects")
            if python_skill is None:
                missing_inputs.append("Python")
            if sql_skill is None:
                missing_inputs.append("SQL")
            if java_skill is None:
                missing_inputs.append("Java")

            if missing_inputs:
                st.warning(f"Please select: {', '.join(missing_inputs)}")
                return

            with st.spinner("Analyzing profile and predicting career..."):
                time.sleep(3)

            user_data = pd.DataFrame(
                [
                    {
                        "Age": age,
                        "GPA": gpa,
                        "Interested Domain": interested_domain,
                        "Projects": projects,
                        "Python": python_skill,
                        "SQL": sql_skill,
                        "Java": java_skill,
                    }
                ]
            )

            encoded_data = encode_user_input(user_data, feature_encoders, model)

            prediction = model.predict(encoded_data)
            predicted_career = target_encoder.inverse_transform(prediction)[0]

            prob_df = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(encoded_data)[0]
                top_idx = np.argsort(proba)[-3:][::-1]
                top_classes = target_encoder.inverse_transform(top_idx)
                top_scores = proba[top_idx]

                prob_df = pd.DataFrame(
                    {
                        "Career": top_classes,
                        "Confidence": [round(float(score), 3) for score in top_scores],
                    }
                )

            st.session_state["prediction_payload"] = {
                "predicted_career": predicted_career,
                "prob_df": prob_df,
            }
            st.session_state["show_prediction_dialog"] = True
            st.rerun()

        if st.session_state.get("show_prediction_dialog"):
            show_prediction_dialog()


if __name__ == "__main__":
    main()
