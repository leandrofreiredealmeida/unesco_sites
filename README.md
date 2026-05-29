# UNESCO World Heritage Sites

Análise e modelagem dos sítios culturais e naturais inscritos na Lista do Patrimônio Mundial da UNESCO.

## Dados

| Arquivo | Descrição |
|---|---|
| `data/raw/whc001.csv` | Dataset oficial da UNESCO — 1.248 sítios, 54 colunas |
| `data/processed/whc_processed.csv` | Dataset pós-EDA — colunas não-inglesas removidas, features derivadas adicionadas |

## Pipeline

### EDA — `notebooks/01_eda.ipynb`

**Limpeza:**
- Remoção de 20 colunas multilíngues (FR, ES, RU, AR, ZH)
- Descarte de `date_end` (100% nulo)

**Feature Engineering:**

| Feature | Origem | Descrição |
|---|---|---|
| `year_inscribed`, `decade_inscribed`, `years_on_list` | `date_inscribed` | Granularidade temporal |
| `n_cultural_criteria`, `n_natural_criteria`, `n_criteria` | `cultural_criteria`, `natural_criteria` (texto) | Contagem por parse de string |
| `latitude`, `longitude` | `coordinates` (string `"lat,lon"`) | Split numérico |
| `has_coordinates` | `latitude`/`longitude` | Flag de completude geográfica |
| `is_endangered` | `danger` (bool) | Alias semântico |
| `description_word_count`, `full_text` | `short_description_en` | Features para pipeline NLP |

**Análise exploratória:** distribuição temporal por décadas, por categoria e região, frequência individual dos critérios de inscrição (C1–C6, N7–N10), dispersão geográfica, sítios em perigo e mapa de correlações entre features numéricas.

## Próximas Fases

- **Clustering geográfico** — DBSCAN / KMeans sobre `latitude`, `longitude`, `area_hectares`
- **Análise temporal** — série histórica de inscrições por `decade_inscribed`, categoria e região
- **Similaridade por NLP** — TF-IDF ou embeddings sobre `full_text` para similaridade cosseno
- **Dashboard** — painel interativo (Plotly / Streamlit) com critérios de inscrição

## Stack

```
pandas · numpy · matplotlib · seaborn · plotly · scikit-learn · scipy · streamlit
```
