from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# KNN avec PCA
def implementation_with_PCA(X_train_scaled,X_test_scaled, y_train, y_test):
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    knn_pca = KNeighborsClassifier(n_neighbors=5)
    knn_pca.fit(X_train_pca, y_train)
    y_pred_pca = knn_pca.predict(X_test_pca)
    accuracy_with_pca = accuracy_score(y_test, y_pred_pca)
    return accuracy_with_pca