# California Housing Price Prediction — Streamlit App

A Streamlit web app that predicts California property prices using a tuned
XGBoost regression model.

**Created by Maulana Imam Rifai | 2026**

---

## 1. Files in this folder

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application |
| `requirements.txt` | Python dependencies |
| `XGBoostModel_CaliforniaHouse.sav` | Trained model (you must copy this from your notebook — see step 2) |

## 2. Get the trained model file

Your notebook already saves the model in the **"Save Machine Learning Model"**
section:

```python
pickle.dump(final_model, open('XGBoostModel_CaliforniaHouse.sav', 'wb'))
```

Run your notebook end-to-end once, then copy the generated
`XGBoostModel_CaliforniaHouse.sav` file into this same folder as `app.py`.

## 3. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## 4. Deploy to Streamlit Community Cloud (free)

1. Create a public GitHub repository containing:
   - `app.py`
   - `requirements.txt`
   - `XGBoostModel_CaliforniaHouse.sav`
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"**, select your repository, branch, and set
   **Main file path** to `app.py`.
4. Click **Deploy**. The app will build automatically and give you a public URL.

## 5. Model input features

| Feature | Type | Description |
|---|---|---|
| `longitude`, `latitude` | float | Property location coordinates |
| `housing_median_age` | int | Median age of houses in the block |
| `total_rooms` | int | Total rooms in the block |
| `total_bedrooms` | int | Total bedrooms in the block |
| `population` | int | Population in the block |
| `households` | int | Number of households in the block |
| `median_income` | float | Median income (tens of thousands USD) |
| `ocean_proximity` | category | INLAND / <1H OCEAN / NEAR OCEAN / NEAR BAY |

Note: the `ISLAND` category was dropped during preprocessing since it had
only 2 records, so it is intentionally excluded from the app's dropdown.
