from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def visualisation_with_PCA(X_test_lda, y_pred_lda, y_test):
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
    plt.savefig('src/models/KNN/results/implementation/result_knn_pca.png', dpi=300, bbox_inches='tight')
    plt.close()

# KNN avec PCA
def implementation_with_PCA(X_train_scaled,X_test_scaled, y_train, y_test):
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    knn_pca = KNeighborsClassifier(n_neighbors=5)
    knn_pca.fit(X_train_pca, y_train)
    y_pred_pca = knn_pca.predict(X_test_pca)

    np.save("y_pred_lda.npy", y_pred_pca)
    visualisation_with_PCA(X_test_pca, y_pred_pca, y_test)

    accuracy_with_pca = accuracy_score(y_test, y_pred_pca)
    return accuracy_with_pca