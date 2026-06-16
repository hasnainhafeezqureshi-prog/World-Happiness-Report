## Project 2: World Happiness Report Analysis

**Dataset:** [World Happiness Report on Kaggle](https://www.kaggle.com/datasets/unsdsn/world-happiness) — download all yearly CSVs (2015–2019)

**What you're building:** A multi-year comparative analysis of global happiness across regions.

---

### Task 1 — Load & Standardize (Pandas)
- Load all 5 CSVs. Column names differ across years — rename them to a consistent schema: `Country`, `Region`, `Happiness_Score`, `GDP`, `Social_Support`, `Health`, `Freedom`, `Corruption`
- Add a `Year` column to each before concatenating them all with `pd.concat()`

### Task 2 — Clean & Reshape (Pandas)
- Drop rows with nulls in key columns
- Use `.melt()` to convert the factor columns (`GDP`, `Social_Support`, `Health`, `Freedom`, `Corruption`) from wide to long format — you'll use this for a grouped Seaborn plot later
- Use `.astype("category")` on the `Region` column

### Task 3 — NumPy Analysis
- Convert `Happiness_Score` to a NumPy array and compute mean, std, and variance per region using axis-based operations
- Normalize the scores using broadcasting: `(scores - mean) / std` — exactly like the normalization example from your handbook
- Use `np.argsort()` to rank countries by score for the most recent year without using `.sort_values()` first

### Task 4 — Visualize (Matplotlib + Seaborn)
- **Stack plot (Matplotlib):** Average scores of the top 5 regions across all 5 years stacked — tests whether you can prep and pass the right data format to `plt.stackplot()`
- **Scatter plot with colormap (Matplotlib):** GDP vs Happiness Score for all countries, colored by score using `cmap='viridis'` and a colorbar
- **Box plot (Seaborn):** `Happiness_Score` by `Region` — one line in Seaborn vs 40 in Matplotlib
- **Bar plot (Seaborn):** Using your melted DataFrame, plot average factor contribution by region using `sns.barplot()` with `hue="variable"` — this is the payoff for doing the melt correctly