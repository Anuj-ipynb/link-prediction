# Viva Defense Presentation Deck: Predicting Resilient Global Grain Trade Corridors
**Domain:** UN SDG 2 (Zero Hunger) | **Methodology:** Link Prediction in Complex Networks

---

## Slide 1: Title & SDG 2 Context
- **Title**: Link Prediction in Global Bilateral Wheat Trade Networks for Food Security Resilience
- **Objective**: Identify latent and missing trade corridors to buffer against food crises and Black Sea bottlenecks.
- **SDG 2 Focus**: Target 2.c — Adopt measures to ensure the proper functioning of food commodity markets.
- **Speaker Notes**: "Good morning respected members of the panel. Today I present our Social Network Analysis project on predicting global grain trade corridors to support UN SDG 2."

---

## Slide 2: Data Source & Network Construction
- **Primary Data**: CEPII BACI Database (HS-100199 Wheat Trade, 2022).
- **Threshold**: Bilateral trade > $500,000 USD.
- **Topology**: Undirected Simple Graph $G = (V, E)$.
- **Network Dimensions**: $N = 164$ countries, $E = 749$ bilateral trade links.
- **Speaker Notes**: "We extracted raw BACI trade data for wheat trade code 100199, filtered out trivial trades below $500k, and constructed an undirected trade network across countries."

---

## Slide 3: 80/20 Edge Split & Spanning-Tree Integrity
- **Challenge**: Standard random edge deletion risks disconnecting peripheral nations and altering the node set.
- **Solution**: Spanning-Tree-Preserving Split.
  - Locked Minimum Spanning Tree (MST) in $G_{train}$.
  - Randomly sampled 20% non-MST edges for $E_{test}$.
- **Negative Sampling**: Sampled equal number of non-edges $(u, v) \notin E$ with 0% ground-truth overlap.
- **Speaker Notes**: "To preserve network connectivity, we used a spanning-tree-preserving split. This ensured G_train retained all nodes without creating artificial isolates."

---

## Slide 4: Proximity Heuristics & Formulations
- **Common Neighbors (CN)**: Count of shared trade partners.
- **Jaccard Coefficient (JC)**: Normalized neighbor overlap ratio.
- **Adamic-Adar Index (AA)**: Log-degree penalized common neighbor score.
- **Preferential Attachment (PA)**: Product of node degrees.
- **Speaker Notes**: "We implemented four classic proximity heuristics on G_train to predict link formation probabilities."

---

## Slide 5: Benchmark Evaluation & Performance Comparison
- **Top Performer**: Adamic-Adar Index (ROC-AUC = 0.7687, AP = 0.7712).
- **Key Insight**: Logarithmic degree penalization ($\frac{1}{\log |\Gamma(z)|}$) suppresses false-positive noise from major trade hubs like USA and FRA.
- **Speaker Notes**: "Adamic-Adar outperformed Preferential Attachment because PA systematically over-predicts connections between mega-hubs regardless of shared trade contexts."

---

## Slide 6: Top-20 Predicted Corridors for Food Security
- **Top Predicted Partnerships**: Inter-continental and South-South trade links.
- **Policy Utility**: Enables international food organizations (FAO, WFP) to incentivize resilient trade agreements before severe climate shocks occur.
- **Speaker Notes**: "Using Adamic-Adar, we extracted the Top-20 unobserved trade links. These represent high-potential bilateral corridors for food supply diversification."

---

## Slide 7: Visualizations & Interactive Graph Artifacts
- **GraphML Export**: Fully annotated with centralities for Cytoscape.
- **PyVis Interactive Explorer**: Color-coded by continent, size-scaled by degree.
- **Degree Distribution & Heatmap**: Demonstrates small-world properties.
- **Speaker Notes**: "We exported interactive PyVis HTML networks and Cytoscape GraphML artifacts with custom node and edge attribute encodings."

---

## Slide 8: Conclusion & Viva Q&A
- **Summary**: Topological link prediction is a robust, data-driven methodology for enhancing global agricultural trade resilience under UN SDG 2.
- **Future Directions**: Incorporating weighted temporal trade dynamics and GNN (Graph Neural Network) embeddings.
- **Speaker Notes**: "Thank you for your time. I am now open to your questions."
