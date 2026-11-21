
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"

df = pd.read_csv(
    RESULTADOS / "jogos_integrados.csv"
)

periodos = {
    "2013-2014": (2013, 2014),
    "2015-2016": (2015, 2016),
    "2017-2018": (2017, 2018)
}

resultados = []

for nome, (inicio, fim) in periodos.items():

    dados = df[
        df["ano_metacritic"].between(inicio, fim)
    ].copy()

    dados["faixa_vendas"] = pd.cut(
        dados["total_sales"],
        bins=[-0.001, 0.1, 0.5, 1, 5, float("inf")],
        labels=[
            "Até 100 mil",
            "100 a 500 mil",
            "500 mil a 1 milhão",
            "1 a 5 milhões",
            "Acima de 5 milhões"
        ]
    )

    contagem = dados["faixa_vendas"].value_counts(
        sort=False
    )

    for faixa, quantidade in contagem.items():
        resultados.append({
            "periodo": nome,
            "faixa_vendas": faixa,
            "quantidade": quantidade,
            "percentual": (
                quantidade / len(dados) * 100
                if len(dados) > 0 else 0
            )
        })

    print(f"\nPERÍODO: {nome}")
    print("Jogos:", len(dados))
    print(
        "Vendas médias:",
        round(dados["total_sales"].mean(), 4),
        "milhões"
    )
    print(
        "Vendas medianas:",
        round(dados["total_sales"].median(), 4),
        "milhões"
    )
    print(
        "Maior venda:",
        round(dados["total_sales"].max(), 4),
        "milhões"
    )

    print("\nDistribuição:")
    print(contagem.to_string())

resumo = pd.DataFrame(resultados)

resumo.to_csv(
    RESULTADOS / "distribuicao_periodos.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("Arquivo gerado: distribuicao_periodos.csv")
print("ANÁLISE CONCLUÍDA!")
