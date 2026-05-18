from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def build_model(random_state: int = RANDOM_STATE) -> Pipeline:
    """
    Build the training pipeline.

    SMOTE is applied only during training. During inference, the fitted
    scaler and classifier are used, but no resampling is performed.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=random_state)),
        ("model", RandomForestClassifier(random_state=random_state)),
    ])


def train_model(X_train, y_train, random_state: int = RANDOM_STATE) -> Pipeline:
    """
    Fit and return the model pipeline.
    """
    pipeline = build_model(random_state=random_state)
    pipeline.fit(X_train, y_train)
    return pipeline
