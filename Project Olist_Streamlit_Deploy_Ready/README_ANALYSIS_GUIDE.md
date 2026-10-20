# Olist E-Commerce Analytics & Machine Learning Project

## Files and what they do

- `Fproject.ipynb` — the original notebook, retained with its existing analysis/model sections, plus an expanded EDA and data-quality section at the end. The CSV loading cell now searches for `Olist_Master_Dataset.csv` in the project folder first.
- `Olist_Master_Dataset.csv` — the master dataset used by the notebook and dashboard project.
- `app.py` — Streamlit dashboard entry point. **Note:** the current code in this file creates sample/simulated records for some dashboard visuals; it is not automatically reading every chart from the CSV.
- `api.py` — FastAPI API code for the project.
- `rf_late_delivery_model.pkl` — saved Random Forest model for late-delivery classification.
- `xgb_churn_model.pkl` — saved XGBoost churn model.
- `requirements.txt` — Python dependencies for the notebook, models, API, and dashboard.
- `README_ANALYSIS_GUIDE.md` — this file, with run instructions and a guide to interpreting analysis outputs.

## Recommended environment

Use Python 3.11 or 3.12. In Windows PowerShell, open this project folder and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.12` is not available, use an installed Python version compatible with the packages.

## Run the full notebook

1. Open the project folder in VS Code or start Jupyter from the folder.
2. Open `Fproject.ipynb`.
3. Select the `.venv` Python kernel.
4. Choose **Restart Kernel and Run All Cells** (in VS Code: notebook toolbar → Run All).
5. Let all cells finish. The dataset is large, and model training may take several minutes.
6. Review the output after each cell. The markdown immediately before each added analysis cell explains what the output means and how to use it.

If the notebook cannot find the dataset, put `Olist_Master_Dataset.csv` in the same folder as `Fproject.ipynb`, which is the default expected layout.

## Display the Streamlit dashboard

In PowerShell, from the project folder with `.venv` activated:

```powershell
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open that URL in your browser. Stop it with `Ctrl+C`.

## Run the API (optional)

Open a second PowerShell terminal in the project folder:

```powershell
uvicorn api:app --reload
```

If the API object in `api.py` has a different name, use the object name defined in that file. FastAPI's interactive documentation is normally at `http://127.0.0.1:8000/docs`.

## Reading the analysis results

- **Shape and structure:** confirms the loaded row/column count and column types.
- **Missing values:** percentages show where missingness is concentrated. Missing delivery dates may represent undelivered/cancelled orders and should be handled according to the specific task.
- **Duplicates and granularity:** repeated order IDs can be legitimate because a master table may combine item, payment, and review records. Do not treat repeated IDs as exact duplicate rows.
- **Outlier screening:** IQR fences flag candidates, not necessarily errors. Compare with the 99th-percentile price cap already present in the original notebook.
- **Class imbalance:** use per-class precision/recall, macro-F1, and confusion matrices in addition to accuracy.
- **Time-series:** check the observed date range and whether the latest period is partial before forecasting.
- **Delivery prediction:** avoid features only known after delivery when predicting lateness in advance; that would leak information into the model.
- **Money totals:** `price` is item price in this master table. A joined table can repeat `payment_value`; validate row grain before summing it.

## Suggested presentation order

1. Dataset overview and business objective.
2. Data-quality findings (missing values, duplicates, outliers).
3. EDA (time trends, categories, geography, payments, reviews, delivery).
4. Existing project visualizations and modelling tasks.
5. Model evaluation and limitations.
6. Business recommendations supported by actual outputs.


## Additional EDA sections added
- df.shape, df.info(), df.describe()
- Missing values and duplicate rows
- Univariate, bivariate, and multivariate analysis
- Numeric feature boxplots and IQR outlier counts
- Correlation matrix and strongest correlation pairs
- Monthly sales and product categories
- Delivery duration and late-delivery rate
- Customer review analysis
- Geographic comparison and customer coordinates
- Cleaning impact before versus after cleaning

Run the notebook from top to bottom to generate the dataset-specific numbers and plots.
