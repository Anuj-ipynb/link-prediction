# Dataset Card: Global Bilateral Wheat Trade Network (UN SDG 2 Benchmark)

## Dataset Overview
- **Domain**: UN Sustainable Development Goal 2 (Zero Hunger) — Global Agricultural Trade Corridors
- **Primary Source**: CEPII BACI International Trade Database (Harmonized System 2022 Nomenclature, Year 2022)
- **Product Filter**: HS-100199 (*Wheat and Meslin, other than durum wheat, other than seed*)
- **Threshold Criterion**: Bilateral annual trade volume exceeding **$500,000 USD** (`v > 500` kUSD)
- **Graph Topology**: Undirected Simple Graph $G = (V, E)$
- **Node Count ($N$)**: 164 sovereign nations
- **Edge Count ($E$)**: 749 bilateral trade corridors

## Schema Documentation

### 1. `data/edges_processed.csv`
| Column | Type | Description |
| :--- | :--- | :--- |
| `source_iso` | String (ISO3) | ISO 3166-1 alpha-3 code for first trading country |
| `target_iso` | String (ISO3) | ISO 3166-1 alpha-3 code for second trading country |
| `trade_value_k_usd` | Float | Aggregated annual bilateral wheat trade in thousand USD |
| `weight` | Float | Log-transformed edge weight (log10(1 + trade_value_k_usd)) |

### 2. `data/nodes_metadata.csv`
| Column | Type | Description |
| :--- | :--- | :--- |
| `node_id` | String (ISO3) | Primary key / ISO 3166-1 alpha-3 country identifier |
| `country_name` | String | Formal sovereign state name |
| `continent` | String | Regional continent categorization |

## Data Collection & Preprocessing Methodology
1. **Extraction**: Raw BACI trade records stream-filtered for commodity classification code `k == 100199`.
2. **Standardization**: UN numerical country codes mapped to ISO3 standard formats via `country_codes_V202601.csv`.
3. **Aggregation**: Directed trade flows ($u \to v$ and $v \to u$) combined into undirected pairwise edges with aggregated trade values.
4. **Pruning**: Isolated singletons removed to yield a single connected global trade component representing major wheat exporting hubs and importing recipient nations.
