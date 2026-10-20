# Olist Project — Notebook Guide

## Main notebook
`Fproject.ipynb` contains the original project cells plus added EDA, plain-Egyptian-Arabic output notes, and a Decision Tree classifier with `plot_tree`.

## Files
- `Fproject.ipynb`: analysis, plots, machine learning and tree visualization.
- `Olist_Master_Dataset.csv`: source dataset.
- `app.py`: Streamlit dashboard.
- `api.py`: FastAPI endpoints.
- `rf_late_delivery_model.pkl`: saved late-delivery Random Forest model (after running notebook export cell).
- `xgb_churn_model.pkl`: saved customer churn classifier (after running notebook export cell).
- `requirements.txt`: Python dependencies.

## Run in VS Code
1. Extract the ZIP and open this folder in VS Code.
2. Create/activate a Python environment and run `pip install -r requirements.txt`.
3. Open `Fproject.ipynb`, choose the Python kernel, and use **Restart Kernel and Run All Cells**.
4. To show the dashboard run `streamlit run app.py`.
5. To start the API run `uvicorn api:app --reload`.

The notebook's comments explain outputs in simple Egyptian Arabic. Metrics and counts are generated from the dataset when the cells run; they are not hard-coded.


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
