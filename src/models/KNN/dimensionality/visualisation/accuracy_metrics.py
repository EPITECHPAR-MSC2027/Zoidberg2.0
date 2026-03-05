import matplotlib.pyplot as plt
import numpy as np

def accuracy_visualisation(methods,accuracies, colors):# Créer la figure avec plusieurs visualisations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Graphique en barres
    ax1 = axes[0]
    bars = ax1.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Comparaison des Accuracy - KNN vs PCA vs LDA', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Ajouter les valeurs sur les barres
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # 2. Graphique en lignes avec marqueurs
    ax2 = axes[1]
    x_pos = np.arange(len(methods))
    ax2.plot(x_pos, accuracies, marker='o', linewidth=2.5, markersize=10, 
            color='#2C3E50', markerfacecolor='#E74C3C', markeredgewidth=2, markeredgecolor='#2C3E50')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods)
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Évolution de la Performance', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Ajouter les valeurs sur les points
    for i, acc in enumerate(accuracies):
        ax2.text(i, acc + 0.03, f'{acc:.4f}', ha='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig('src/models/KNN/results/implementation/comparison_knn_pca_lda.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique sauvegardé : comparison_knn_pca_lda.png")