from sklearn.datasets        import load_breast_cancer
from sklearn.preprocessing   import StandardScaler
from sklearn.cluster         import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics         import silhouette_score

def main():
    # ─── Load the Wisconsin breast–cancer data ───────────────
    bc = load_breast_cancer()
    X = bc.data.astype(float)

    # ─── Scale features ──────────────────────────────────────
    X_scaled = StandardScaler().fit_transform(X)

    # ─── K-Means: try k=2…5 ──────────────────────────────────
    print("=== KMeans Silhouettes (breast_cancer) ===")
    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X_scaled)
        score  = silhouette_score(X_scaled, labels)
        print(f"  k = {k:>1} → silhouette = {score:.4f}")

    # ─── Agglomerative (k=2…5) ──────────────────────────────
    print("\n=== Agglomerative (Ward linkage) ===")
    for k in [2, 3, 4, 5]:
        agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = agg.fit_predict(X_scaled)
        score  = silhouette_score(X_scaled, labels)
        print(f"  k = {k:>1} → silhouette = {score:.4f}")

    # ─── DBSCAN (ε and min_samples chosen heuristically) ────
    print("\n=== DBSCAN ===")
    db = DBSCAN(eps=1.2, min_samples=5).fit(X_scaled)
    unique_labels = set(db.labels_)
    n_clusters    = len(unique_labels) - ( -1 in unique_labels )
    print(f"  clusters found = {n_clusters}, labels = {sorted(unique_labels)}")

if __name__ == "__main__":
    main()
