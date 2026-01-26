import numpy as np
from sklearn.preprocessing import StandardScaler

def dataset_to_arrays(dataset_split, image_size=(128, 128)):
    X = []
    y = []
    for example in dataset_split:
        # Convertir l'image PIL en grayscale
        img = example['image'].convert('L')
        
        # ✅ REDIMENSIONNER à la même taille
        img_resized = img.resize(image_size)
        
        # Convertir en array numpy et aplatir
        img_array = np.array(img_resized, dtype='float64').flatten()
        X.append(img_array)
        y.append(example['label'])
    
    return np.array(X), np.array(y)

def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled
