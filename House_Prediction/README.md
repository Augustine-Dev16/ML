# House Price Prediction using XGBoost

This project demonstrates how to use the XGBoost framework to predict housing prices using the California Housing dataset. It’s a classic regression problem where the goal is to estimate median house values based on features like location, population, and room counts.

## Dataset

We use the **California Housing dataset**, available directly from `sklearn.datasets`.

- Features include:
  - `MedInc`: Median income in block group
  - `HouseAge`: Median house age
  - `AveRooms`: Average number of rooms
  - `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`
- Target variable:
  - `MedHouseVal`: Median house value (in $100,000s)

## Objective

Train a gradient boosting regression model to accurately predict house prices and interpret feature importance.

## Workflow

### 1. Load and Inspect Data
- Use `fetch_california_housing()` from `sklearn.datasets`
- Convert features into a DataFrame for readability

### 2. Preprocessing
- Split data into training and testing sets (80/20 split)

### 3. Modeling
- Use `XGBRegressor` from the `xgboost` library
- Parameters used:
  - `objective='reg:squarederror'`
  - `n_estimators=100`
  - `learning_rate=0.1`
- Fit model on training data

### 4. Evaluation
- Predict on test set
- Compute:
  - **Root Mean Squared Error (RMSE)**
  - **R² score**
- Visualize feature importance

## Results

| Metric | Value |
|--------|-------|
| RMSE   | ~0.50 |
| R²     | ~0.80 |

(Exact values may vary slightly depending on system and train-test split)

## Requirements

Install the necessary libraries:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```

## How to Run

Run the Python script:

```bash
python xgboost_house_price.py
```

Or use a Jupyter Notebook for interactive execution and plotting.

## File Structure

```
.
├── xgboost_house_price.py
├── README.md
```

## License

This project is released for educational use under the MIT License.
