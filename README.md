
# Santander Bank Customer Segmentation — Cluster Analysis

> **End-to-end cluster analysis comparing Ward's Hierarchical Cluster Analysis (HCA) and K-Means on 4,521 Santander Bank loan customers. Cophenetic correlation 0.91 confirms excellent HCA fit. K-Means achieves a higher Silhouette Score (0.3105 vs 0.2368) and is recommended for production deployment.**

---

## Business Context

You work in analytics at **Santander Bank**, an international banking company. The task is to cluster loan customers in their US market to identify distinct segments that can inform:
- Targeted marketing campaigns
- Personalized product offerings
- Credit risk assessment and monitoring strategies

---

## Dataset — `bank_cluster.csv`

**4,521 customers · 8 features · 0 missing values**

| Variable | Type | Description | Range |
|---|---|---|---|
| `age` | Numeric | Age of customer (years) | 19 – 95 |
| `default` | Binary | Credit in default: 1 = yes, 0 = no | 0, 1 |
| `balance` | Numeric | Average yearly account balance ($) | −$3,313 – $98,417 |
| `housing` | Binary | Has housing loan: 1 = yes, 0 = no | 0, 1 |
| `loan` | Binary | Has personal loan: 1 = yes, 0 = no | 0, 1 |
| `duration` | Numeric | Last contact duration (seconds) | 4 – 3,025 |
| `campaign` | Numeric | Number of campaign contacts | 1 – 63 |
| `previous` | Numeric | Contacts before this campaign | 0 – 275 |

> **Note:** `default` is held out as the external validation label (used for Adjusted Rand Index only). It is not included in the clustering feature matrix.

---

## Descriptive Statistics

| Statistic | age | balance | housing | loan | duration | campaign | previous |
|---|---|---|---|---|---|---|---|
| Mean | 41.17 | $1,422.66 | 0.57 | 0.15 | 263.96 s | 2.79 | 0.58 |
| Std | 10.58 | $3,009.64 | 0.50 | 0.36 | 259.86 s | 3.11 | 2.30 |
| Min | 19 | −$3,313 | 0 | 0 | 4 s | 1 | 0 |
| Median | 39 | $444 | 1 | 0 | 185 s | 2 | 0 |
| Max | 95 | $98,417 | 1 | 1 | 3,025 s | 63 | 275 |

**Key observations:**
- `balance` has extreme right skew — the median ($444) is far below the mean ($1,423), driven by high-balance outliers above $50,000.
- `default` is highly imbalanced: only 76 customers (1.7%) have defaulted. This makes it valuable as an external validator but impractical as a clustering feature.
- `campaign` and `previous` are right-skewed with most customers contacted only 1–3 times.

---

## Figure 1 — Feature Distributions by Default Status

<img width="1737" height="779" alt="image" src="https://github.com/user-attachments/assets/d57386e8-4d21-446c-a552-b9aa4e89ee47" />


**What the chart shows:**  
An 8-panel histogram grid displaying the distribution of every feature in the dataset. Bars are colored by `default` status — blue for non-defaulters (default = 0) and red for defaulters (default = 1). The `default` panel itself shows a bar chart of class counts.

**Key insights:**
- The `default` panel confirms severe class imbalance: **4,445 non-defaulters (98.3%)** vs. **76 defaulters (1.7%)**. This is why ARI values are expected to be near zero — unsupervised clusters are unlikely to align with such a rare minority class.
- Defaulters (`red`) tend to appear across all age and balance ranges, confirming that default status is not cleanly separable by any single feature.
- `balance` shows a broad right-skewed distribution dominated by low-balance customers, with a long tail of high-balance customers appearing almost exclusively in the non-default group.
- `housing` and `loan` show roughly equal splits for non-defaulters but the defaulter group exhibits notably higher rates of housing loan ownership.

---

## Preprocessing

### Q1(a) — Feature/Target Split

```
X shape : (4521, 7)   — features for clustering
y shape : (4521,)     — default status (held out)

Features: ['age', 'balance', 'housing', 'loan', 'duration', 'campaign', 'previous']

Target distribution:
  default = 0 : 4,445 customers (98.3%)
  default = 1 :    76 customers  (1.7%)
```

### Q1(b) — MinMaxScaler Normalization (for HCA)

MinMaxScaler rescales every feature to the range [0, 1] using the formula `x' = (x − min) / (max − min)`. Without normalization, `balance` (range: ~$100,000) would completely dominate Euclidean distance calculations relative to binary features like `housing` (range: 0–1).

```
After normalization — first 5 rows (sample):
      age  balance  housing  loan  duration  campaign  previous
  0.2353   0.0445      1.0   0.0    0.1566    0.0000      0.00
  0.3824   0.0484      1.0   0.0    0.0291    0.0000      0.00
  ...
All column min = 0.0, max = 1.0 ✓
```

**For K-Means**, StandardScaler was used instead (zero mean, unit variance), which is preferred when clusters are expected to be approximately spherical in standardized space.

---

## Ward's Hierarchical Cluster Analysis

### Q2(a) — Linkage Matrix

```
Linkage matrix shape: (4520, 4)
First 5 merge steps:
   Cluster 1   Cluster 2   Distance   New Size
      970.0      3423.0     0.0003        2.0
     1940.0      2035.0     0.0003        2.0
     1417.0      2117.0     0.0004        2.0
      123.0       932.0     0.0005        2.0
       94.0      3989.0     0.0007        2.0
```

Each row represents one merge step. The very small initial distances (0.0003–0.0007) confirm that the first merges combine nearly identical customers. Ward's method merges the pair that produces the smallest increase in total within-cluster variance.

### Q2(b) — Cophenetic Correlation

```
Cophenetic Correlation Coefficient : 0.9144
Model fit quality                  : Excellent
```

**What the Cophenetic Correlation measures:** It compares the pairwise distances between all customers in the original feature space against the distances at which they are merged in the dendrogram. A value of 1.0 would mean the dendrogram perfectly represents the original distances. A value of **0.9144** means the hierarchical tree preserves 91.4% of the original distance information — an excellent fit. Values above 0.90 are considered excellent; values below 0.70 suggest the hierarchy poorly represents the data.

---

## Figure 2 — HCA Silhouette Score vs. Number of Clusters

<img width="1192" height="532" alt="image" src="https://github.com/user-attachments/assets/1596b1e2-2854-42aa-a1f3-d638b161e7f5" />


**What the chart shows:**  
A line plot of the Mean Silhouette Score for k = 2 through 25 clusters using Ward's HCA (AgglomerativeClustering). Each point represents the silhouette score at that number of clusters. The red dashed vertical line marks the optimal k identified by the highest silhouette score. The red dot highlights the peak.

**Key results:**
```
Optimal k (highest silhouette): k = 4
Silhouette score at k = 4    : 0.7627
```

**Why k = 25 was used despite k = 4 being optimal:**  
The silhouette analysis identifies k = 4 as statistically optimal. However, the assignment specified k = 25 to enable more granular segmentation for downstream marketing use cases where sub-segment targeting is more actionable than four broad groups. Both results are reported.

**Reading the chart:** The silhouette score generally decreases as k increases — a common pattern because more clusters mean smaller, less cohesive groups. The sharp drop from k = 4 onward confirms that four clusters is the natural elbow point in the HCA silhouette curve.

---

## Figure 3 — Ward's HCA Dendrogram (Full, Truncated to 5 Levels)

<img width="1522" height="642" alt="image" src="https://github.com/user-attachments/assets/1bace2d9-a942-40e3-a497-2835d932eb3e" />


**What the chart shows:**  
A dendrogram (tree diagram) representing the hierarchical merging process of Ward's HCA. The x-axis shows customer indices or cluster sizes (in parentheses when truncated). The y-axis shows the Ward Linkage Distance — the increase in total within-cluster variance caused by each merge. The red dashed horizontal line marks the cut point for k = 25 clusters.

**How to read it:** Each horizontal line at height `h` represents two clusters being merged at cost `h`. Cutting the dendrogram horizontally at any height produces the number of clusters equal to the number of vertical lines the cut intersects. The higher the cut, the fewer and larger the resulting clusters.

**Key observations:**
- The dendrogram is truncated to 5 levels (showing the top portion of the tree) to maintain readability with 4,521 data points.
- The relatively flat base of the tree, with many merges occurring at low distances, confirms that the data contains tight local groupings before they consolidate into broader clusters.
- The cut at k = 25 falls relatively low in the tree — confirming that 25 clusters represents a fine-grained segmentation.

---

## Figure 4 — Ward's HCA Dendrogram (Zoomed)

<img width="1522" height="642" alt="image" src="https://github.com/user-attachments/assets/c0e7dbad-867c-43fd-91ad-8acf397c86d1" />


**What the chart shows:**  
The same dendrogram zoomed in to focus on the lower portion of the tree — the region where detailed cluster structure is most visible. The y-axis is capped at 1.5× the cut height for k = 25.

**Why this is useful:** The full dendrogram compresses most of the meaningful structure at the bottom into an unreadable range because the top-level merges occur at very high distances. The zoomed view reveals the sub-cluster structure near the leaves more clearly, showing which customer groups are most similar and are merged earliest.

---

## HCA Cluster Size Distribution

### Q2(d) — Cluster Assignments at k = 25

```
k_val_HCA = 25
Total customers assigned: 4,521

Cluster  Size  |  Cluster  Size  |  Cluster  Size
      1    71  |        9   361  |       17   141
      2    12  |       10    32  |       18   332
      3   314  |       11    30  |       19   254
      4    74  |       12   152  |       20   174
      5   462  |       13   496  |       21    25
      6    61  |       14   273  |       22   207
      7   293  |       15   413  |       23   125
      8    29  |       16    30  |       24   (cont.)
```

---

## Figure 5 — HCA Cluster Sizes (k = 25)

<img width="1522" height="422" alt="image" src="https://github.com/user-attachments/assets/767ca8c7-0553-40a1-baba-3150e0fc64a3" />


**What the chart shows:**  
A bar chart displaying the number of customers in each of the 25 HCA clusters. Each bar represents one cluster, with the count labeled above.

**Key observations:**
- Cluster sizes are **highly unequal**, ranging from just 12 customers (Cluster 2) to 496 customers (Cluster 13). This is expected behavior for Ward's HCA — the algorithm optimizes variance reduction, not cluster size balance.
- The concentration of most customers in a handful of large clusters (5, 9, 13, 15, 18) while several clusters have fewer than 30 customers reflects the natural density variation in the data.
- Very small clusters (2, 8, 10, 11, 16, 21) likely represent outlier sub-groups or niche customer profiles that are meaningfully distinct but numerically rare.
- From a business perspective, large clusters should be prioritized for marketing investment, while small clusters may warrant case-by-case investigation for niche product development.

---

## HCA Cluster Centroids

### Q3(a) — Ward's HCA Cluster Centroids (Original Scale, first 4 shown)

| HCA Cluster | Age | Default | Balance | Housing | Loan | Duration (s) | Campaign | Previous |
|---|---|---|---|---|---|---|---|---|
| 1 | 74.2 | 0.000 | $2,811 | 0.0 | 0.0 | 330 | 2.3 | 0.86 |
| 2 | 55.9 | 0.000 | $2,319 | 0.0 | 0.0 | 393 | 1.2 | 11.2 |
| 3 | 57.0 | 0.003 | $2,616 | 0.0 | 0.0 | 227 | 2.2 | 0.27 |
| 4 | 30.3 | 0.000 | $208 | 1.0 | 0.0 | … | … | … |

### Q3(b) — Cluster 1 Description

```
Cluster 1 centroid profile:
  age          74.169
  default       0.000
  balance    $2,811.38
  housing        0.000   (no housing loans)
  loan           0.000   (no personal loans)
  duration     329.563 s
  campaign       2.254
  previous       0.859

Cluster 1 size: 71 customers
```

**Business interpretation:** Cluster 1 consists of **71 older customers** (average age ~74) with positive average balances of $2,811, no credit defaults, and no active loans. Their extended campaign contact duration (330 seconds, ~5.5 minutes) suggests engagement but potentially slow decision-making typical of older demographic groups. The complete absence of housing and personal loans combined with a healthy positive balance identifies this group as **asset-rich, credit-averse retirees or near-retirees**. Santander should target this segment with wealth management services, fixed-income investment products, and estate planning offerings rather than credit products.

---

## K-Means Cluster Analysis

### Q4(a) — Grid Search Results (k = 2 to 25)

```
k    Inertia (WSS)   Silhouette Score
2       26,935.18         0.2194
3       23,107.10         0.2626
4       20,635.51         0.2756
5       18,541.39         0.2866
6       16,357.83         0.2958
7       14,183.56         0.3105  ← OPTIMAL (highest silhouette)
8       12,825.91         0.3014
9       11,908.44         0.2705
10      11,252.06         0.2803
…
25       …                …
```

---

## Figure 6 — K-Means Optimization: Elbow Method and Silhouette Score

<img width="1522" height="557" alt="image" src="https://github.com/user-attachments/assets/f8396aef-1072-4297-916d-0ae614b73493" />


**What the chart shows:**  
Two side-by-side line plots for k = 2 through 25:
- **Left panel (Elbow / WSS):** Inertia (within-cluster sum of squares) plotted against the number of clusters k. Color: chocolate brown.
- **Right panel (Silhouette):** Mean Silhouette Score plotted against k. The red dashed vertical line marks the optimal k. Color: green.

**Reading the Elbow plot:**  
Inertia always decreases as k increases — adding more clusters always reduces within-cluster variance. The "elbow" is the point where the rate of decrease slows significantly, suggesting diminishing returns from adding more clusters. The curve bends noticeably around k = 4–7 before flattening, consistent with the silhouette finding.

**Reading the Silhouette plot:**  
```
Optimal k by silhouette: k = 7
Silhouette score at k = 7: 0.3105
```
Unlike the elbow plot, the silhouette score has a clear maximum at k = 7 before declining. This peak at 0.3105 identifies 7 as the statistically optimal number of clusters for K-Means on this dataset.

---

## K-Means Results

### Q5(a) — Cluster Size Distribution at k = 7

```
Optimal k (k_val_KM): 7

Cluster  Size
      0  1,719   (38.0%)
      1  1,375   (30.4%)
      2    621   (13.7%)
      3    154    (3.4%)
      4    196    (4.3%)
      5    141    (3.1%)
      6    315    (7.0%)
```

K-Means produces more balanced cluster sizes than HCA, with the largest cluster (0: 1,719 customers) being approximately 12× the smallest (5: 141 customers). This is a more operationally useful distribution than HCA's range of 12–496 customers.

### Q5(b) — K-Means Validation Metrics

```
WSS (Inertia)                : 14,183.5185
Mean Silhouette Score        :      0.3105
Adjusted Rand Index (ARI)   :      0.0005
```

---

## K-Means Cluster Centroids

### Q6(a) — K-Means Cluster Centroids (Original Scale)

| KM Cluster | Age | Balance | Housing | Loan | Duration (s) | Campaign | Previous |
|---|---|---|---|---|---|---|---|
| Cluster 0 | 39.5 | $921 | 1.000 | 0.000 | 205 | 2.30 | 0.27 |
| Cluster 1 | 43.3 | $1,122 | 0.000 | 0.000 | 209 | 2.43 | 0.25 |
| Cluster 2 | 41.0 | $779 | 0.586 | 1.000 | 229 | 2.48 | 0.25 |
| Cluster 3 | 43.7 | $13,218 | 0.506 | 0.039 | 218 | 2.48 | 0.54 |

**Cluster 0** — Mid-30s customers with moderate balance (~$921), all holding housing loans, no personal loans. Likely the primary homeowner segment. Stable, engaged, and moderate risk.

**Cluster 1** — Mid-40s customers, no housing or personal loans, similar contact duration. The bank's core no-debt stable customer segment — strong savings profile.

**Cluster 2** — Early 40s, moderate balance (~$779), mix of housing loans, and **all have personal loans** (`loan = 1.000`). This group carries the highest personal loan burden and warrants targeted loan management or upsell of consolidation products.

**Cluster 3** — High-balance customers with average balance over **$13,218** — the bank's premium segment. Roughly equal housing loan ownership, almost no personal loans. Target with wealth management and premium banking services.

---

## Figure 7 — K-Means Cluster Centroid Heatmap

<img width="1153" height="422" alt="image" src="https://github.com/user-attachments/assets/7477b918-6171-4560-9125-af9529820818" />

**What the chart shows:**  
A heatmap of the 7 K-Means cluster centroids in **standardized space** (z-scores). Each row is a cluster (C0–C6) and each column is a feature. Red cells indicate values significantly above average; blue cells indicate values significantly below average; white/neutral indicates near-average values. Annotated with the exact standardized centroid value in each cell.

**How to read it:**  
- **Deep red** in `balance` for one cluster (Cluster 3) confirms that this group's average balance is several standard deviations above the mean — the high-balance premium segment.
- **Blue in `loan`** for most clusters but **red for Cluster 2** confirms that Cluster 2 exclusively carries personal loans.
- **Red in `housing`** for Cluster 0 confirms the housing loan segment.
- **Blue in `age`** for younger clusters and **red for `age`** in the older clusters confirms natural age-based segmentation.

**Business value:** This visualization allows Santander's marketing team to immediately identify which features drive each cluster's distinctiveness without reading through a full centroids table.

---

## Model Comparison

### Q7(a) — Validation Metrics Side by Side

```
==========================================================
     VALIDATION METRICS COMPARISON
==========================================================
Metric                          HCA (Ward's)    K-Means
----------------------------------------------------------
WSS (Within-Cluster Sum of Sq)       57.99    14,183.52
Mean Silhouette Score                0.2368       0.3105
Adjusted Rand Index (ARI)            0.0002       0.0005
Optimal k                                25            7
==========================================================
```

**Interpreting the WSS difference:**  
HCA's WSS (57.99) appears dramatically lower than K-Means (14,183.52) because they are computed in different feature spaces — HCA used MinMax-normalized data (values 0–1), while K-Means used StandardScaler data (values typically −3 to +3). These WSS values are **not directly comparable** across models; they are only meaningful within each model's own optimization context.

---

## Figure 8 — Model Validation Metrics Comparison

<img width="1522" height="557" alt="image" src="https://github.com/user-attachments/assets/c9a3dd87-7380-484a-a0be-3d8527457932" />

**What the chart shows:**  
Three side-by-side bar charts comparing Ward's HCA (blue) and K-Means (green) on three validation dimensions:
1. **WSS** — Within-cluster sum of squares (lower = tighter clusters within each model's scaled space)
2. **Silhouette** — Mean silhouette score (higher = more cohesive and well-separated clusters)
3. **ARI** — Adjusted Rand Index vs. `default` label (higher = better alignment with known default status)

**Key takeaways:**
- **Silhouette** is the most meaningful comparison: K-Means (0.3105) outperforms HCA (0.2368). A silhouette score of 0.3105 indicates reasonably well-separated clusters, while 0.2368 indicates weaker structure.
- **ARI** is near zero for both models (0.0002 vs. 0.0005), which is expected given the extreme class imbalance in `default` (98.3% vs. 1.7%). Neither model's clusters systematically align with the rare default event — this is not a model failure but a reflection of the imbalance.

---

## Final Recommendation

### Q7(b) — K-Means Recommended for Production

**K-Means is the preferred model** for Santander's customer segmentation for three reasons:

**1. Higher Silhouette Score (0.3105 vs. 0.2368)**  
K-Means produces tighter, more cohesive clusters with better inter-cluster separation. This means customer segments are more meaningfully distinct — essential for actionable marketing targeting.

**2. Scalability for Production**  
Ward's HCA requires O(n²) pairwise distance computation across all 4,521 customers and cannot update when new customers are added without recomputing the entire dendrogram. K-Means assigns new customers to existing clusters in constant time using centroid distance — practical for a bank with a continuously changing customer base.

**3. Interpretable Centroids in Business Units**  
K-Means centroids are explicit coordinate vectors in the original feature space (after inverse scaling), making them directly readable by marketing and risk teams. HCA does not produce centroids — cluster profiles must be computed after the fact via `groupby`.

> HCA's Cophenetic Correlation of 0.91 confirms that the dendrogram accurately represents the data's hierarchical structure. However, this is a property of the tree's fidelity — not a measure of how useful the resulting clusters are for downstream business decisions.

---

## Analytics Pipeline Summary

```
Data Loading (4,521 customers, 8 features)
    ↓
Q0: Explore — DataFrame info, descriptive statistics, feature distributions
    ↓
Q1: Split y (default) and X (7 features)
    MinMaxScaler  → X          [for HCA]
    StandardScaler → X_scaled  [for K-Means]
    ↓
Q2: Ward's HCA on MinMax-normalized X
    ├─ Linkage matrix (4520 × 4) computed
    ├─ Cophenetic r = 0.9144  ← Excellent
    ├─ Silhouette k=2–25 → optimal k = 4 (stat.) / k = 25 (assignment)
    ├─ Dendrogram (full + zoomed)
    └─ WSS=57.99, Silhouette=0.2368, ARI=0.0002
    ↓
Q3: HCA Centroids (original scale) + Cluster 1 description
    ↓
Q4: K-Means optimization (Elbow + Silhouette k=2–25)
    └─ grid initialized → loop k=2–25 → grid_df
    ↓
Q5: Fit K-Means (k=7, seed=123)
    └─ WSS=14,183.52, Silhouette=0.3105, ARI=0.0005
    ↓
Q6: K-Means Centroids (original scale via inverse_transform) + Cluster 2 description
    ↓
Q7: Compare → Recommend K-Means
```

---

## Figures Reference

| Figure | File | Section | What It Shows |
|---|---|---|---|
| Fig 1 | `feature_distributions.png` | Q0 EDA | 8-panel histograms colored by default status |
| Fig 2 | `hca_silhouette.png` | Q2(c) | HCA Silhouette score k=2–25, optimal k=4 marked |
| Fig 3 | `hca_dendrogram.png` | Q2(e) | Full truncated Ward's dendrogram, cut at k=25 |
| Fig 4 | `hca_dendrogram_zoom.png` | Q2(e) | Zoomed dendrogram showing lower-level merge structure |
| Fig 5 | `hca_cluster_sizes.png` | Q3(a) | Bar chart of customer count per HCA cluster (k=25) |
| Fig 6 | `kmeans_optimization.png` | Q4(a) | Elbow (WSS) + Silhouette side-by-side, optimal k=7 |
| Fig 7 | `kmeans_centroids_heatmap.png` | Q6(a) | Heatmap of 7 K-Means standardized centroids |
| Fig 8 | `model_comparison.png` | Q7(a) | Three-panel bar chart: WSS, Silhouette, ARI — HCA vs K-Means |

---

## Tech Stack

```
Python 3.10
├── pandas          — data loading, groupby centroids, DataFrame operations
├── NumPy           — WSS computation, array operations
├── scikit-learn    — MinMaxScaler, StandardScaler, KMeans,
│                     AgglomerativeClustering, silhouette_score,
│                     adjusted_rand_score
├── scipy           — linkage, dendrogram, fcluster, cophenet, pdist
├── matplotlib      — all charts, dendrograms, multi-panel figures
└── seaborn         — heatmap of K-Means centroids, themed styling
```

---

## How to Run

**Google Colab (recommended)**
```python
# Runtime → Run all
# Dataset loads automatically from GitHub raw URL — no file upload needed
url = 'https://raw.githubusercontent.com/CHill-MSU/INFO583/main/bank_cluster.csv'
```

**Local Jupyter**
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
jupyter notebook Santander_Bank_Customer_Segmentation___Cluster_Analysis.ipynb
```

---

## Skills Demonstrated

`Hierarchical Cluster Analysis` `K-Means Clustering` `Ward's Linkage` `Cophenetic Correlation` `Silhouette Analysis` `Elbow Method` `MinMaxScaler` `StandardScaler` `Adjusted Rand Index` `Dendrogram` `Cluster Validation` `Business Analytics` `Customer Segmentation` `Python` `scikit-learn` `scipy` `seaborn`

---

## Author

**Aketch Adhiambo Okoth**  
MS Business Analytics — Montclair State University   
[LinkedIn](https://linkedin.com/in/your-profile) · [Portfolio](https://your-portfolio-url.com)
