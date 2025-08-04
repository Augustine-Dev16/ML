# CO₂ Emissions Prediction using Multiple Linear Regression

This project demonstrates how to build and visualize a multiple linear regression model to predict vehicle CO₂ emissions based on engine size and fuel consumption data.

## Dataset

The dataset is publicly available and hosted by IBM:

[FuelConsumptionCo2.csv](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv)

It contains information on fuel consumption, engine size, and corresponding CO₂ emissions for a variety of car models.

## Features Used

**Input Features:**
- `ENGINESIZE`: Size of the vehicle’s engine (liters)
- `FUELCONSUMPTION_COMB_MPG`: Fuel consumption in miles per gallon (combined cycle)

**Target Variable:**
- `CO2EMISSIONS`: Carbon dioxide emissions in grams per kilometer

## Workflow

### 1. Data Preparation
- Load data using `pandas`
- Drop non-numeric and irrelevant columns such as:
  - `MAKE`, `MODEL`, `FUELTYPE`, etc.
- Remove collinear features based on correlation analysis
- Normalize the data using `StandardScaler` for consistent feature scaling

### 2. Model Training
- Split the dataset into training and testing sets (80/20 split)
- Train a multiple linear regression model using two independent variables:
  - `ENGINESIZE`
  - `FUELCONSUMPTION_COMB_MPG`
- Output model coefficients and intercept
- Convert standardized coefficients back to original scale for interpretation

### 3. Visualization
- Create a 3D scatter plot showing the regression plane and data points
- Generate 2D regression lines for:
  - `CO2EMISSIONS` vs `ENGINESIZE`
  - `CO2EMISSIONS` vs `FUELCONSUMPTION_COMB_MPG`
- Compare model fit using both training and test data

### 4. Additional Models
- Retrain and evaluate two simple linear regression models using:
  - Only `ENGINESIZE`
  - Only `FUELCONSUMPTION_COMB_MPG`
- Visualize each model with line-of-best-fit and scatterplots

## Key Outputs

- Coefficients (standardized and original)
- 3D regression surface
- 2D regression lines
- Evaluation of the model's performance via visual inspection of predictions

## Requirements

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## How to Run

This script can be run in any Python environment. If using a Jupyter Notebook, uncomment the line:

```python
%matplotlib inline
```

to enable inline plotting.

## Notes

- Standardization is essential for comparing features on different scales
- Coefficients are translated back to original units for clearer interpretation
- Visualization provides intuitive understanding of model performance

## File Structure

```
.
├── CO2.py (or .ipynb)
├── README.md
```

## License

This project is for educational purposes and is released under the MIT License.

