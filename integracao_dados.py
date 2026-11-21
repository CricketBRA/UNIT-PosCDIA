
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"

# Carregar os arquivos preparados
metacritic = pd.read_csv(
    RESULTADOS / "metacritic_preparado.csv"
)

vendas = pd.read_csv(
    RESULTADOS / "vendas_preparadas.csv"
)

# Padronizar nomes das plataformas
mapa_plataformas = {
    "PlayStation": "PS",
    "PlayStation 2": "PS2",
    "PlayStation 3": "PS3",
    "PlayStation 4": "PS4",
    "PlayStation 5": "PS5",
    "PlayStation Portable": "PSP",
    "PlayStation Vita": "PSV",
    "Xbox": "XB",
    "Xbox 360": "X360",
    "Xbox One": "XOne",
    "Xbox Series X": "XS",
    "Nintendo Switch": "NS",
    "Nintendo Switch 2": "NS2",
    "Nintendo DS": "DS",
    "Nintendo 3DS": "3DS",
    "Nintendo 64": "N64",
    "Nintendo GameCube": "GC",
    "Nintendo Wii": "Wii",
    "Nintendo Wii U": "WiiU",
    "Game Boy Advance": "GBA",
    "Game Boy Color": "GBC",
    "PC": "PC",
}

metacritic["plataforma_padrao"] = (
    metacritic["platform"]
    .replace(mapa_plataformas)
)

vendas["plataforma_padrao"] = vendas["console"]

# Criar identificadores para acompanhar correspondências
metacritic["id_metacritic"] = range(len(metacritic))
vendas["id_vendas"] = range(len(vendas))

# Relacionar por nome e plataforma
integrado = pd.merge(
    metacritic,
    vendas,
    on=["nome_normalizado", "plataforma_padrao"],
    how="inner",
    suffixes=("_metacritic", "_vendas")
)

# Comparar os anos de lançamento
integrado["diferenca_anos"] = (
    integrado["ano_metacritic"]
    - integrado["ano_vendas"]
).abs()

# Manter correspondências com diferença de até 1 ano
integrado = integrado[
    integrado["diferenca_anos"] <= 1
].copy()

# Identificar possíveis ambiguidades
repeticoes_meta = integrado["id_metacritic"].duplicated(
    keep=False
)

repeticoes_vendas = integrado["id_vendas"].duplicated(
    keep=False
)

integrado["correspondencia_ambigua"] = (
    repeticoes_meta | repeticoes_vendas
)

# Separar correspondências únicas das ambíguas
confirmadas = integrado[
    ~integrado["correspondencia_ambigua"]
].copy()

ambiguas = integrado[
    integrado["correspondencia_ambigua"]
].copy()

# Salvar os resultados
confirmadas.to_csv(
    RESULTADOS / "jogos_integrados.csv",
    index=False,
    encoding="utf-8-sig"
)

ambiguas.to_csv(
    RESULTADOS / "correspondencias_ambiguas.csv",
    index=False,
    encoding="utf-8-sig"
)

print("=" * 60)
print("INTEGRAÇÃO DOS DATASETS")
print("=" * 60)

print("\nRegistros Metacritic:", len(metacritic))
print("Registros de vendas:", len(vendas))

print("\nCorrespondências com diferença de até 1 ano:")
print(len(integrado))

print("\nCorrespondências únicas:")
print(len(confirmadas))

print("\nCorrespondências ambíguas para revisão:")
print(len(ambiguas))

print("\nJogos integrados com notas e vendas:")
print(
    confirmadas[
        [
            "name",
            "platform",
            "metascore",
            "user_score",
            "total_sales"
        ]
    ].head(10).to_string(index=False)
)

print("\nArquivos gerados:")
print("- jogos_integrados.csv")
print("- correspondencias_ambiguas.csv")

print("\nINTEGRAÇÃO INICIAL CONCLUÍDA!")
