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

### Modelagem — `notebooks/02_modeling.ipynb`

**Fase 1 — Análise Temporal:**
Série histórica de inscrições agrupadas por décadas, com linha de tendência linear. Composição por categoria (Cultural / Natural / Mixed) em visão absoluta e proporcional. Heatmap de inscrições por região × década.

**Fase 2 — Similaridade por NLP:**
Vetorização de `full_text` com `TfidfVectorizer` (`ngram_range=(1,2)`, `sublinear_tf=True`, `max_features=8_000`, `min_df=2`). Consulta de similaridade entre sítios via `linear_kernel`. Similaridade média entre regiões via vetor TF-IDF médio + `cosine_similarity`, exibida em heatmap.

**Fase 3 — Clustering Geográfico (DBSCAN):**

| Decisão | Valor |
|---|---|
| Métrica | `haversine` com coordenadas em radianos |
| Algoritmo | `ball_tree` |
| eps | 0,05 rad ≈ 318 km (selecionado via gráfico k-distâncias, k=5) |
| min_samples | 5 |

Gráfico k-distâncias expõe o cotovelo natural dos dados e justifica o eps escolhido. Resultado exibido em scatter plot lon × lat colorido por cluster, com legenda de tamanho e região predominante dos cinco maiores clusters.

> Variantes multivariada (lat/lon + `year_inscribed` + `n_criteria` + `category`) e temática NLP foram avaliadas e descartadas por não produzirem estrutura interpretável.

**Fase 4 — Classificação Temática:**

12 categorias temáticas criadas para oferecer uma taxonomia mais evocativa que Cultural / Natural / Mixed:

| # | Categoria |
|---|---|
| 1 | Ruínas e sítios arqueológicos |
| 2 | Centros urbanos vivos |
| 3 | Conjuntos religiosos e sagrados |
| 4 | Fortificações e sistemas defensivos |
| 5 | Paisagens culturais e parques humanizados |
| 6 | Palácios e sedes de poder político |
| 7 | Necrópoles e espaços funerários |
| 8 | Engenharia e infraestrutura histórica |
| 9 | Conjuntos rurais e aldeias |
| 10 | Sítios industriais e tecnológicos |
| 11 | Arte rupestre e inscrições primitivas |
| 12 | Monumentos comemorativos e memoriais |

**Método — prototype matching:** cada categoria é representada por um texto-protótipo em inglês transformado pelo `TfidfVectorizer` já ajustado na Fase 2. A similaridade cosseno entre os 1.248 sítios e os 12 protótipos produz uma matriz (1 248 × 12); a categoria primária é o `argmax` da linha; atribuições multi-label usam limiar de 0,05.

**Visualizações:**
- Barras horizontais: distribuição de sítios por categoria primária
- Top 5 sítios por categoria com scores de similaridade (validação qualitativa)
- Scatter plot geográfico colorido pelas 12 categorias
- Heatmap de co-ocorrência multi-label entre categorias
- Stacked bar: categorias temáticas × classificação UNESCO (Cultural / Natural / Mixed)