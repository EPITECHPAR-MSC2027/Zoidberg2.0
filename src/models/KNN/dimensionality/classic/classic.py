from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# KNN sans réduction
def classic_KNN(X_train_scaled,y_train,X_test_scaled, y_test, ):
    print("\n1. KNN sans réduction de dimensionnalité...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracy_without_reduction = accuracy_score(y_test, y_pred)
    return f" {accuracy_without_reduction:.4f}"