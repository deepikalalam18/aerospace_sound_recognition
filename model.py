from sklearn.ensemble import RandomForestClassifier

def build_model():
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    return model
