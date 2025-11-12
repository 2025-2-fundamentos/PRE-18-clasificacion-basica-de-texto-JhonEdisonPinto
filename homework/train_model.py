# homework/train_model.py
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar datos
dataframe = pd.read_csv(
    "files/input/sentences.csv.zip",
    index_col=False,
    compression="zip",
)

print("Columnas del dataset:", dataframe.columns.tolist())

# Preparar datos
X = dataframe['phrase'].fillna('')
y = dataframe['target']

print(f"Ejemplos de frases: {X[:3].tolist()}")
print(f"Targets únicos: {y.unique()}")

# Vectorizar texto
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8
)

X_vectorized = vectorizer.fit_transform(X)
print(f"Datos vectorizados: {X_vectorized.shape}")

# Entrenar modelo
clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=30,
    min_samples_split=5
)

clf.fit(X_vectorized, y)

# Calcular accuracy
accuracy = accuracy_score(y, clf.predict(X_vectorized))
print(f"Accuracy: {accuracy:.4f}")

# GUARDAR CON LOS NOMBRES EXACTOS QUE EL TEST ESPERA
with open("homework/clf.pickle", "wb") as file:  # ¡.pickle no .pkl!
    pickle.dump(clf, file)

with open("homework/vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("✅ Modelo guardado como: homework/clf.pickle")
print("✅ Vectorizer guardado como: homework/vectorizer.pkl")
print(f"✅ Accuracy: {accuracy:.4f}")