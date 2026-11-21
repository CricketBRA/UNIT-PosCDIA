
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"

arquivos = {
    "Regressão": "metricas_regressao.csv",
    "Random Forest": "metricas_random_forest.csv",
    "Random Forest Log": "metricas_random_forest_log.csv",
    "Modelo sem ano": "metricas_sem_ano.csv",
    "Impacto das notas": "comparacao_impacto_notas.csv",
    "Validação temporal": "validacao_temporal_com_referencia.csv",
    "Sensibilidade": "teste_sensibilidade_integracao.csv"
}

print("=" * 65)
print("CONSOLIDAÇÃO DOS RESULTADOS")
print("=" * 65)

for nome, arquivo in arquivos.items():

    caminho = RESULTADOS / arquivo

    print(f"\n{'-' * 65}")
    print(nome.upper())
    print(f"{'-' * 65}")

    if caminho.exists():
        tabela = pd.read_csv(caminho)
        print(tabela.round(4).to_string(index=False))
    else:
        print(f"Arquivo não encontrado: {arquivo}")

print("\n" + "=" * 65)
print("CONSOLIDAÇÃO CONCLUÍDA!")
