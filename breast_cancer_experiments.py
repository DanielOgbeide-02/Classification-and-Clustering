from sklearn.preprocessing import StandardScaler
from sklearn.datasets      import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LogisticRegression
from sklearn.svm           import SVC
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.metrics       import classification_report, roc_auc_score

def main():
    bc = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        bc.data, bc.target, test_size=0.2,
        random_state=0, stratify=bc.target
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # compare three classifiers
    for name, cls in [
        ("LogisticRegression", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ("SVM", SVC()),
        ("DecisionTree", DecisionTreeClassifier(random_state=42))
    ]:
        if name in ("LogisticRegression", "SVM"):
            cls.fit(X_train_scaled, y_train)
            y_pred = cls.predict(X_test_scaled)
        else:
            cls.fit(X_train, y_train)
            y_pred = cls.predict(X_test)
        print(f"\n{name}:\n", classification_report(y_test, y_pred, digits=4))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_rf = rf.predict(X_test)
    print("\nRandomForest:\n", classification_report(y_test, y_rf, digits=4))

    # ROC-AUC
    proba = rf.predict_proba(X_test)[:,1]
    print("RF ROC AUC:", roc_auc_score(y_test, proba))

if __name__ == "__main__":
    main()
