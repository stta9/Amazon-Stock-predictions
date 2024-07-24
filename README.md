# Amazon Stock Price Prediction

## Installation

To run this project, you need to install the required packages. You can install them using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Dataset

The dataset used in this project is Amazon.csv.xls, which contains historical stock data. The dataset has the following columns:

- Date: Date of the stock data
- Open: Opening price
- High: Highest price
- Low: Lowest price
- Close: Closing price
- Adj Close: Adjusted closing price
- Volume: Trading volume

## Data Preprocessing
1. Loading the dataset: The dataset is loaded into a pandas DataFrame.
2. Date conversion: The Date column is converted to datetime format.
3. Setting index: The Date column is set as the index.
4. Handling missing values: Any missing values in the dataset are handled (in this case, the dataset does not have missing values).

```python
df = pd.read_csv('/content/Amazon.csv.xls')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.isnull().sum()
```

## Modeling and testing
The following models are used for stock price prediction: Random Forest Regressor and Gradient Boosting Regressor.RMSE is calculated for performance evaluation for both of the models.

```python
# Prepare features and target
X = df.drop(columns=['Close'])
y = df['Close']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))

# Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
print("Gradient Boosting RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_gb)))
```

## Hyperparameter Tuning
```python
# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}

# Initialize and fit GridSearchCV
gb_model = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best parameters and performance
best_params = grid_search.best_params_
print("Best parameters: ", best_params)

best_gb_model = GradientBoostingRegressor(**best_params, random_state=42)
best_gb_model.fit(X_train, y_train)
y_pred_best_gb = best_gb_model.predict(X_test)
print("Tuned Gradient Boosting RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_best_gb)))
```

## Final decision making

```python
plt.figure(figsize=(15, 5))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred_rf, label='Random Forest prediction', alpha=0.5)
plt.plot(y_test.index, y_pred_gb, label='Gradient Boosting prediction', alpha=0.5)
plt.legend()
plt.show()
```

## Conclusion
The Random Forest model has a lower RMSE compared to the Gradient Boosting model. This indicates that, on average, the predictions made by the Random Forest model are closer to the actual stock prices than those made by the Gradient Boosting model.

