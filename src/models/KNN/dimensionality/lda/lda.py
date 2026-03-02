# KNN avec LDA
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
def visualisation_with_LDA(X_test_lda, y_pred_lda, y_test):
    plt.figure(figsize=(10, 7))
    
    colors = ['green', 'red', 'blue']
    labels = ['NORMAL', 'BACTERIA', 'VIRUS']
    
    # Points corrects = pleins
    for i, label in enumerate(labels):
        correct = (y_test == i) & (y_pred_lda == i)
        plt.scatter(X_test_lda[correct, 0], X_test_lda[correct, 1],
                   color=colors[i], label=f'{label} ✓',
                   s=150, alpha=0.8, edgecolors='black', linewidth=1.5)
    
    # Points incorrects = contour rouge épais
    misclassified = y_pred_lda != y_test
    if np.any(misclassified):
        for i in np.unique(y_test[misclassified]):
            mask = misclassified & (y_test == i)
            plt.scatter(X_test_lda[mask, 0], X_test_lda[mask, 1],
                       color=colors[i], s=150, alpha=0.8,
                       edgecolors='red', linewidth=3,
                       label=f'{labels[i]} ✗')
    
    plt.xlabel('LDA Component 1', fontweight='bold')
    plt.ylabel('LDA Component 2', fontweight='bold')
    accuracy = np.mean(y_pred_lda == y_test)
    plt.title(f'KNN + LDA (Accuracy: {accuracy:.1%})', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Sauvegarder l'image
    plt.savefig('src/models/KNN/results/implementation/result_knn_lda.png', dpi=300, bbox_inches='tight')
    plt.close()

def implementation_with_LDA(y_train, X_train_scaled, X_test_scaled, y_test):
    n_classes = len(np.unique(y_train))
    lda = LinearDiscriminantAnalysis(n_components=min(2, n_classes-1))
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)
    X_test_lda = lda.transform(X_test_scaled)
    
    knn_lda = KNeighborsClassifier(n_neighbors=5)
    knn_lda.fit(X_train_lda, y_train)

    y_pred_lda = knn_lda.predict(X_test_lda)
    np.save("y_pred_lda.npy", y_pred_lda)
    visualisation_with_LDA(X_test_lda, y_pred_lda, y_test)
    
    accuracy_with_lda = accuracy_score(y_test, y_pred_lda)
    
    return accuracy_with_lda