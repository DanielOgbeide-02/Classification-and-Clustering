from sklearn.datasets        import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import SVC
from sklearn.tree            import DecisionTreeClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import classification_report

def main():
    # 1. Load and split
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target,
        test_size=0.2,
        random_state=0,
        stratify=data.target
    )

    # 2. Scale for LR & SVM
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 3. Define models
    models = {
        "LogisticRegression":   LogisticRegression(max_iter=1000, solver="lbfgs"),
        "SVM":                  SVC(),
        "DecisionTree (rs=0)":  DecisionTreeClassifier(random_state=0),
        "DecisionTree (rs=42)": DecisionTreeClassifier(random_state=42),
    }

    # 4. Train & evaluate
    for name, model in models.items():
        if name in ("LogisticRegression", "SVM"):
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, digits=4))

if __name__ == "__main__":
    main()
