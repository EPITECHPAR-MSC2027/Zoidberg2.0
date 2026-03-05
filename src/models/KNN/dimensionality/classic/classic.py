import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# KNN sans réduction
def classic_KNN(X_train_scaled,y_train,X_test_scaled, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)

    for i, (pred, real) in enumerate(zip(y_pred, y_test)):
        statut = "✅" if pred == real else "❌"
        print(f"Image {i+1} : Prédit={pred}, Réel={real} {statut}")

    accuracy_with_lda = accuracy_score(y_test, y_pred)
    print(f"\nPrécision finale : {accuracy_with_lda * 100:.2f}%")

    np.save("y_pred_lda.npy", y_pred)

    accuracy_without_reduction = accuracy_score(y_test, y_pred)
    return accuracy_without_reduction