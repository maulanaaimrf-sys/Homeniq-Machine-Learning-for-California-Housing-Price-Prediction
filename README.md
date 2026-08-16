# Homeniq – California Housing Price Prediction

Machine learning project to estimate California residential property values, built as a decision-support tool for appraisers, lenders, and property owners.

## Background

Homeniq is a (fictional) independent property valuation firm in California. Traditional appraisal relies heavily on manual analysis, which is slow and can be inconsistent. This project builds a regression model that produces fast, consistent, data-driven property value estimates to support — not replace — certified appraisers.

## Dataset

Based on the classic California housing census dataset.

- **14,448 rows** (13,763 after cleaning), **10 columns**
- 9 numerical features, 1 categorical feature (`ocean_proximity`)

| Column | Description |
|---|---|
| `longitude` / `latitude` | Geographic coordinates |
| `housing_median_age` | Median age of houses in the area |
| `total_rooms` | Total rooms in the district |
| `total_bedrooms` | Total bedrooms in the district |
| `population` | Population in the district |
| `households` | Number of households in the district |
| `median_income` | Median household income |
| `ocean_proximity` | Distance category from the ocean (e.g. `INLAND`, `NEAR BAY`, `<1H OCEAN`) |
| `median_house_value` | Median house value (target variable) |

## Approach

1. **EDA** – distribution, correlation, and geographic analysis of house prices.
2. **Data Preprocessing**
   - Imputed 137 missing values in `total_bedrooms` with the median.
   - Removed capped/outlier records at `median_house_value = 500,001` (~4.7% of data).
   - Removed the near-empty `ISLAND` category from `ocean_proximity`.
   - One-Hot Encoding for the categorical feature; Robust Scaling for numerical features.
3. **Modeling** – compared Linear Regression, Decision Tree, KNN, Random Forest, and XGBoost via cross-validation, then benchmarked the top two (Random Forest vs. XGBoost) on the test set.
4. **Hyperparameter Tuning** – `RandomizedSearchCV` on XGBoost.
5. **Evaluation** – RMSE, MAE, MAPE.

## Results

**XGBoost** was the best-performing model.

| Metric | Before Tuning | After Tuning |
|---|---|---|
| RMSE | 44,682.81 | **43,072.63** |
| MAE | 30,038.59 | **28,961.86** |
| MAPE | 17.71% | **17.00%** |

**Top features:** `ocean_proximity_INLAND` (44.3% importance) and `median_income` (17.4%) are the strongest drivers of predicted house value.

## Key Limitations

- Data reflects the 1990 census and does not represent current market conditions.
- Records above $500,000 were removed during preprocessing, so the model is not reliable for high-value properties (recommended guardrail: use manual appraisal above $500K).
- Prediction error increases for higher-priced properties.

## Recommendations

- Retrain/re-evaluate the model periodically (e.g. every 6 months).
- Use GridSearch for further hyperparameter optimization and consider iterative feature selection.
- Enrich the dataset with newer data and additional property/location features (e.g. lot size, amenities, GIS data).
- Treat model output as decision support, not a replacement for certified appraisers.

## Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `XGBoost` · `seaborn` / `matplotlib` · `geopandas`

## Project Structure

```
├── Caps3__California_Housing_Price.ipynb   # Full analysis & modeling notebook
├── XGBoostModel_CaliforniaHouse.sav        # Saved tuned XGBoost model (pickle)
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib geopandas category_encoders statsmodels
jupyter notebook Caps3__California_Housing_Price.ipynb
```

Load the saved model for inference:

```python
import pickle
model = pickle.load(open('XGBoostModel_CaliforniaHouse.sav', 'rb'))
prediction = model.predict(X_new)
```
