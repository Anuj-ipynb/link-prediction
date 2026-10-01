# 🌾 Predicting Resilient Global Grain Trade Corridors
> **Link Prediction in Complex Bilateral Trade Networks for UN SDG 2 (Zero Hunger)**

[![UN SDG 2](https://img.shields.io/badge/UN%20SDG-2%20Zero%20Hunger-emerald.svg)](https://sdgs.un.org/goals/goal2)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.0%2B-orange.svg)](https://networkx.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Project Overview
Global food security relies heavily on the structural resilience of international agricultural trade corridors. Geopolitical crises, extreme weather events, and maritime supply chain bottlenecks pose severe threats to staple cereal supply lines—particularly wheat.

Aligning with **UN Sustainable Development Goal 2 (Zero Hunger)**, this project constructs a complex network representation of global bilateral wheat trade ($N=164$ sovereign nations, $E=749$ active trade links) using **CEPII BACI 2022 International Trade Data**. We implement and benchmark four topological proximity heuristics (**Common Neighbors**, **Jaccard Coefficient**, **Adamic-Adar Index**, and **Preferential Attachment**) to predict latent and resilient grain trade corridors.

---

## 🚀 Key Features & Methodology

- **Data Streaming & Processing**: Extracted commodity classification `HS-100199` (*Wheat & Meslin*) with a trade volume threshold $v > \$500,000\text{ USD}$.
- **Spanning-Tree Component Preservation**: Implemented an 80/20 train/test split utilizing a Minimum Spanning Tree (MST) lock on $G_{train}$ to prevent graph fragmentation and isolate zero nodes.
- **Strict Negative Sampling**: Sampled $|E_{test\_neg}| = 149$ non-existent trade edges with 0% ground-truth overlap.
- **Topological Heuristics Benchmark**: Evaluated performance across ROC-AUC, Average Precision (AP), Precision@K, Recall, and optimal F1-Score.
- **Top-20 SDG 2 Policy Recommendations**: Scored unobserved node pairs to identify high-probability bilateral trade bridges for food security shock absorption.
- **Interactive & GraphML Visualizations**: Exported Cytoscape-ready GraphML networks and dynamic PyVis HTML web explorers.

---

## 📊 Quantitative Benchmark Results

Evaluated on an 80/20 train-test edge split ($E_{train}=600$, $E_{test}=149$, $E_{neg}=149$):

| Heuristic Model | ROC-AUC | Average Precision (AP) | Precision@10 | Precision@20 | Precision@50 | Recall (Max F1) | Optimal F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Preferential Attachment** | **0.8862** | **0.8342** | 0.8000 | 0.8500 | 0.8400 | 0.8993 | **0.8375** |
| **Adamic-Adar Index** | 0.7687 | 0.7712 | 0.8000 | 0.9000 | 0.9000 | 0.7383 | 0.7458 |
| **Common Neighbors** | 0.7496 | 0.7352 | 0.8000 | 0.9000 | 0.8600 | 0.8121 | 0.7014 |
| **Jaccard Coefficient** | 0.6096 | 0.5414 | 0.5000 | 0.5000 | 0.5800 | 0.8121 | 0.7014 |

---

## 🗂️ Project Structure & Deliverables

```
├── link_prediction_pipeline.ipynb   # Executable Jupyter Notebook with saved cell outputs
├── build_full_project.py           # End-to-end Python pipeline execution script
├── data/
│   ├── edges_processed.csv         # Processed bilateral trade edge list
│   ├── nodes_metadata.csv          # Sovereign country metadata & continent labels
│   └── dataset_card.md             # Complete dataset package documentation
├── exports/
│   ├── grain_network.graphml       # Annotated network graph for Cytoscape visualization
│   └── grain_network.html          # Interactive PyVis HTML network explorer
├── visualizations/
│   ├── degree_distribution.png     # Degree histogram, KDE, and log-log rank plot
│   ├── adjacency_heatmap.png       # Global trade network adjacency matrix heatmap
│   └── model_comparison_curves.png # Comparative ROC and Precision-Recall curves
├── report/
│   ├── top_20_predicted_corridors.csv  # Predicted corridors enriched with SDG 2 rationales
│   ├── evaluation_report.md        # Formal academic evaluation report & Cytoscape guide
│   └── presentation_deck.md        # Viva defense presentation deck with speaker notes
├── .gitignore                      # Git ignore rules for raw BACI datasets (>350MB)
└── README.md                       # Repository documentation
```

---

## 💻 Installation & How to Run

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/Anuj-ipynb/link-prediction.git
cd link-prediction

pip install pandas numpy networkx matplotlib seaborn scikit-learn pyvis jupyter tabulate
```

### 2. Run Pipeline Script
To regenerate all datasets, metrics, reports, plots, and execute the notebook:
```bash
python build_full_project.py
```

### 3. Open Interactive Notebook
```bash
jupyter notebook link_prediction_pipeline.ipynb
```

---

## 🎨 Cytoscape Visual Styling Guide
1. Import `exports/grain_network.graphml` into **Cytoscape** (`File -> Import -> Network from File`).
2. Set Layout to **Prefuse Force Directed Layout**.
3. **Node Color**: Map `continent` column using *Discrete Mapping* (Africa `#ef4444`, Asia `#f59e0b`, Europe `#10b981`, North America `#3b82f6`, South America `#8b5cf6`, Oceania `#ec4899`).
4. **Node Size**: Map `degree_centrality` column using *Continuous Mapping* (Range 20–80).
5. **Edge Type**: Map `edge_type` column using *Discrete Mapping* (`train` `#475569`, `test_ground_truth` `#38bdf8`, `predicted_top20` dashed `#ec4899`).

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
