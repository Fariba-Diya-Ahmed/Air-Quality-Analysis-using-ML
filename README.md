# Air-Quality-Analysis-using-ML

A machine learning-based Air Quality Index prediction system built from real pollution data for **Sylhet, Bangladesh**. The project analyzes how pollutant concentrations, time of day, and seasonal patterns affect AQI - and predicts air quality with health status classification.

---

## Project Description Summary
The goal is to predict the **Air Quality Index (AQI)** for Sylhet city and classify air quality into health status categories (**Good, Moderate, Unhealthy, Harmful, Very Harmful, Severe**) based on pollutant features such as PM2.5, PM10, NO₂, SO₂, CO, and Ozone.

The dataset was explored visually and statistically to understand AQI distribution, seasonal trends, hourly patterns, and pollutant correlations. A **K-Nearest Neighbors (KNN) Regressor** was trained with optimal K selection using the Elbow Method. The final output maps predicted AQI values to health status categories using a threshold-based classification function.

---

## Dataset
- **Source:** AQI Bangladesh - Kaggle
- **Focus City:** Sylhet
- **Period:** Last 6 Months
Dataset is too large to upload on GitHub (3M+ rows).
- Download from Google Drive: [AQI Dataset](https://drive.google.com/file/d/1jwC7rTkybqeG25i0TZ3hY_RrkfSjh2tL/view?usp=sharing)

### Key Columns
| Column | Description |
|--------|-------------|
| `datetime` | Timestamp of the record |
| `aqi` | Air Quality Index (target variable) |
| `pm2_5` | Fine particulate matter (PM2.5) |
| `pm10` | Coarse particulate matter (PM10) |
| `nitrogen_dioxide` | NO₂ concentration |
| `sulphur_dioxide` | SO₂ concentration |
| `carbon_monoxide` | CO concentration |
| `ozone` | Ozone (O₃) concentration |
| `month` | Extracted month (time feature) |
| `hour` | Extracted hour (time feature) |

---

## Key Code Elements & Workflows

### 1. Data Preprocessing
- Converted `datetime` column and filtered records to the **last 6 months**
- Filtered dataset to **Sylhet city only**
- Extracted `month` and `hour` as time-based features
- Dropped irrelevant columns: `city_id`, `city_name`, `lat`, `lon`, `carbon_dioxide`
- Removed rows with null AQI values

```python
df['datetime'] = pd.to_datetime(df['datetime'])
df = df[df['datetime'] >= df['datetime'].max() - pd.DateOffset(months=6)]
df = df[df['city_name'] == 'Sylhet'].copy()
df['month'] = df['datetime'].dt.month
df['hour'] = df['datetime'].dt.hour
```

---

### 2. Exploratory Data Analysis (EDA)

**AQI Distribution: Histogram + KDE**
Plotted AQI frequency distribution with mean and median reference lines to understand data spread and central tendency.

**AQI Outliers: Boxplot**
Visualized spread and outliers in AQI values using a horizontal boxplot.

**Correlation Heatmap**
Computed feature correlation matrix with `RdBu` colormap to identify which pollutants most strongly correlate with AQI.

**Seasonal AQI Trend: Line Plot**
Mapped months to seasons (Winter, Pleasant, Extreme Hot, Rainy) and plotted monthly AQI trends to reveal seasonal air quality patterns.

| Season | Months |
|--------|--------|
| Winter | Nov – Feb |
| Pleasant | Mar |
| Extreme Hot | Apr – Jun |
| Rainy | Jul – Oct |

**Hourly AQI Trend: Line Plot**
Plotted AQI variation across 24 hours with highlighted Morning, Noon, and Night zones to reveal daily pollution patterns.

**Feature vs AQI: Scatter Plots**
Generated 6 scatter plots (PM2.5, PM10, NO₂, SO₂, CO, Ozone vs AQI) to visualize individual pollutant relationships with air quality.

---

### 3. Feature Selection & Train-Test Split

**Input Features (X):** `pm2_5`, `pm10`, `nitrogen_dioxide`, `sulphur_dioxide`, `carbon_monoxide`, `ozone`

**Target Variable (y):** `aqi` (continuous - regression task)

**Split Ratio:** 80% Train / 20% Test (`random_state=42`)

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

> ⚠️ **Feature Scaling Applied** - KNN is a distance-based algorithm, so `StandardScaler` was applied to normalize all input features before training.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

### 4. Best K Selection: Elbow Method
RMSE was computed for K values from 1 to 20. The K with the lowest RMSE was selected as the optimal neighbor count.

```python
for k in range(1, 21):
    knn_temp = KNeighborsRegressor(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    pred_temp = knn_temp.predict(X_test_scaled)
    rmse_list.append(np.sqrt(mean_squared_error(y_test, pred_temp)))

best_k = list(k_range)[np.argmin(rmse_list)]
```

---

### 5. Model Training - KNN Regression

| Model | Type | Library |
|-------|------|---------|
| K-Nearest Neighbors Regressor | Instance-based, Non-parametric | `sklearn.neighbors` |

```python
knn = KNeighborsRegressor(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
```

---

### 6. AQI Status Classification
A threshold function maps predicted AQI values to interpretable health status labels:

```python
def get_status(aqi):
    if aqi <= 50:    return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Harmful"
    elif aqi <= 300: return "Very Harmful"
    else:            return "Severe"
```

Applied to both actual and predicted AQI values to produce a side-by-side status comparison table.

---

## Model Evaluation Metrics

| Metric | KNN Regressor |
|--------|--------------|
| Best K | 4 |
| RMSE | 14.0119 |
| MAE | 9.7958 |
| R² Score | 0.7514 |


---

## Visualization Outputs Summary

| # | Plot | Purpose |
|---|------|---------|
| 1 | AQI Distribution Histogram | Frequency spread + mean/median |
| 2 | AQI Boxplot | Outlier detection |
| 3 | Correlation Heatmap | Feature-AQI relationship strength |
| 4 | Seasonal AQI Line Plot | Monthly & seasonal trends |
| 5 | Hourly AQI Line Plot | Daily pollution pattern |
| 6 | Feature vs AQI Scatter Plots | Per-pollutant impact |
| 7 | Elbow Method Plot | Best K selection |
| 8 | Actual vs Predicted Scatter | Model accuracy check |
| 9 | Residual Plot | Bias and error distribution |

---

## Installation & Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Project
1. Clone the repository:
```bash
git clone https://github.com/Fariba-Diya-Ahmed/Air-Quality-Analysis-using-ML
.git
cd Air-Quality-Analysis-using-ML

```

2. Upload the dataset to your Google Drive at:
```
MyDrive/AI_project/aqi_dataset/AQI Bangladesh.csv
```

3. Open and run the notebook:

[AQI_Prediction_ML.ipynb](https://colab.research.google.com/drive/1Dt_zJe5hFttM_Jhvuuf2xUTTO7N7xEAT?usp=sharing)


The notebook will load the dataset, filter for Sylhet, perform EDA, find the best K, train the KNN model, and display all visualizations sequentially.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `matplotlib` | Plot rendering |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | Model training, scaling, evaluation |

---

## Results

- **Model:** K-Nearest Neighbors Regressor
- **Prediction Task:** AQI regression → Health status classification
- **Top Predictors:** PM2.5 and PM10 showed the strongest correlation with AQI
- **Key Insight:** AQI in Sylhet peaks during **Winter months** and follows clear **hourly patterns** throughout the day

This project demonstrates an end-to-end machine learning pipeline for real-world urban air quality prediction, combining regression modeling with interpretable health status output for practical environmental monitoring.

