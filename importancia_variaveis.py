
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import permutation_importance

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"

df = pd.read_csv(
    RESULTADOS / "jogos_integrados.csv"
)

treino = df[df["ano_metacritic"] <= 2015].copy()
teste = df[df["ano_metacritic"] >= 2016].copy()

variaveis = [
    "metascore",
    "user_score",
    "plataforma_padrao",
    "genre_metacritic",
    "ano_metacritic"
]

X_treino = treino[variaveis]
y_treino = treino["total_sales"]

X_teste = teste[variaveis]
y_teste = teste["total_sales"]

preprocessamento = ColumnTransformer(
    transformers=[
        (
            "numericas",
            SimpleImputer(strategy="median"),
            [
                "metascore",
                "user_score",
                "ano_metacritic"
            ]
        ),
        (
            "categoricas",
            OneHotEncoder(handle_unknown="ignore"),
            [
                "plataforma_padrao",
                "genre_metacritic"
            ]
        )
    ]
)

floresta = Pipeline(
    steps=[
        ("preprocessamento", preprocessamento),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

modelo = TransformedTargetRegressor(
    regressor=floresta,
    func=np.log1p,
    inverse_func=np.expm1
)

print("Treinando o modelo...")
modelo.fit(X_treino, y_treino)

print("Calculando importância das variáveis...")

resultado = permutation_importance(
    modelo,
    X_teste,
    y_teste,
    scoring="neg_mean_absolute_error",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importancias = pd.DataFrame({
    "variavel": variaveis,
    "importancia_media": resultado.importances_mean,
    "desvio_padrao": resultado.importances_std
})

importancias = importancias.sort_values(
    "importancia_media",
    ascending=False
)

print("\n" + "=" * 65)
print("IMPORTÂNCIA DAS VARIÁVEIS - RANDOM FOREST LOG")
print("=" * 65)

print(importancias.round(5).to_string(index=False))

importancias.to_csv(
    RESULTADOS / "importancia_variaveis.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(10, 6))

plt.barh(
    importancias["variavel"],
    importancias["importancia_media"],
    xerr=importancias["desvio_padrao"]
)

plt.axvline(0, color="black", linewidth=1)

plt.xlabel(
    "Aumento do MAE ao embaralhar a variável (milhões)"
)
plt.ylabel("Variável")
plt.title("Importância das Variáveis - Random Forest Log")

plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(
    RESULTADOS / "importancia_variaveis.png",
    dpi=300
)

plt.close()

print("\nArquivos gerados:")
print("- importancia_variaveis.csv")
print("- importancia_variaveis.png")

print("\nANÁLISE CONCLUÍDA!")
