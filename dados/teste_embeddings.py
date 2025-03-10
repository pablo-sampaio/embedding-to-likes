import pandas as pd
import sklearn as skl
from sentence_transformers import SentenceTransformer

# 1. Pegar o modelo para testar
type_model = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(type_model)

# 2. Pegar as sentenças (nesse caso, no Post-filtrado)
file_path = 'Post-filtrado.xlsx'
file_path_features = "Embeddings_Feature_allMini.xlsx"
coluna_1 = "Texto"

# coluna_2 = "Curtida"  // Analisar como usar

rf = pd.read_excel(file_path)
rf = rf.dropna(subset=[coluna_1])

sentences = rf[coluna_1].tolist()

# 3. Calcular os embeddings das sentenças
embeddings = model.encode(sentences)
df = pd.DataFrame(embeddings)

df.columns = [f'x{i+1}' for i in range(df.shape[1])]
df.to_excel(file_path_features, index=False, startcol=4)

