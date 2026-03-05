# KNN avec LDA
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

def implementation_with_LDA(y_train,X_train_scaled,X_test_scaled,y_test):
    n_classes = len(np.unique(y_train))
    lda = LinearDiscriminantAnalysis(n_components=min(2, n_classes-1))
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)
    X_test_lda = lda.transform(X_test_scaled)

    knn_lda = KNeighborsClassifier(n_neighbors=5)
    knn_lda.fit(X_train_lda, y_train)
    y_pred_lda = knn_lda.predict(X_test_lda)
    accuracy_with_lda = accuracy_score(y_test, y_pred_lda)
    return f" {accuracy_with_lda:.4f}"