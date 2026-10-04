# Predicting House Prices Using Linear Regression

A supervised machine learning project that uses housing and neighbourhood characteristics to predict the median house value of a suburb, built with Python and scikit-learn in Google Colab.

## Business Problem

A real estate analyst wants to understand how neighbourhood characteristics relate to house prices, and to build a model that estimates property values from available housing data.

## Objective

Develop a regression model that uses housing and neighbourhood features to predict the median house value for a given suburb.

- **Target variable:** `MEDV`, the median owner-occupied home value, recorded in thousands of US dollars
- **Machine learning task:** Supervised learning (regression)
- **Algorithm:** Linear Regression

## Dataset

The project uses the Boston housing dataset (`boston.csv`), which contains 13 features describing suburbs and their housing, plus the target `MEDV`. Features used in the analysis include:

| Feature | Meaning |
|---------|---------|
| `CRIM` | Crime rate |
| `ZN` | Proportion of residential land zoned for large lots |
| `CHAS` | Whether the suburb borders the Charles River |
| `NOX` | Nitric oxide concentration |
| `RM` | Average number of rooms per dwelling |
| `DIS` | Distance to employment centres |
| `PTRATIO` | Pupil-teacher ratio |
| `LSTAT` | Percentage of lower-status population |

## Project Workflow

1. Import libraries
2. Load and preview the dataset
3. Explore the data (`info()`, `describe()`, missing value check)
4. Visualise the relationship between number of rooms (`RM`) and house price (`MEDV`)
5. Separate features (`X`) and target (`y`)
6. Split the data into 80% training and 20% testing sets (`random_state=42`)
7. Train a `LinearRegression` model
8. Make predictions on the test set
9. Evaluate the model with MAE, MSE, RMSE and R²
10. Visualise actual vs predicted prices
11. Examine the model coefficients

## Model

The model estimates the median house value as a linear combination of all 13 features:

```
MEDV = β0 + β1·CRIM + β2·ZN + ... + β13·LSTAT
```

## Results

| Metric | Value | What it means |
|--------|-------|---------------|
| MAE | 3.19 | Predictions are off by about $3,190 on average |
| RMSE | 4.93 | Larger errors are penalised more; being higher than MAE shows some predictions are further off than others |
| R² | 0.67 | The model explains roughly 67% of the variation in house values in the test set |

Note: R² of 0.67 does not mean 67% of individual predictions are correct. It describes how much of the variation in prices the model captures.

### Key Findings

- Average number of rooms (`RM`) shows a positive, roughly linear relationship with house price.
- In the actual vs predicted plot, most points sit close to the line of perfect prediction, so most predictions are reasonably accurate.

### Most Influential Features

| Feature | Coefficient | Interpretation (all else equal) |
|---------|-------------|---------------------------------|
| `NOX` | -17.20 | Higher nitric oxide concentration is associated with lower predicted values |
| `RM` | +4.44 | One extra average room is associated with about $4,440 higher predicted value |
| `CHAS` | +2.78 | Bordering the Charles River is associated with about $2,780 higher predicted value |
| `DIS` | -1.45 | Greater distance is associated with lower predicted values |
| `PTRATIO` | -0.92 | A higher pupil-teacher ratio is associated with about $920 lower predicted value |
| `LSTAT` | -0.51 | A higher percentage of lower-status population is associated with lower predicted values |

## Technologies Used

- Python
- pandas and NumPy
- seaborn and Matplotlib
- scikit-learn
- Google Colab

## How to Run

1. Clone this repository or open `Predicting_House_Prices_using_Linear_Regression.ipynb` in Google Colab.
2. Upload `boston.csv` to your Colab session (the notebook reads it from `/content/boston.csv`).
3. Run all cells from top to bottom.

## Limitations and Next Steps

- Linear regression assumes straight-line relationships, which may not hold for every feature.
- The model leaves about 33% of the variation in prices unexplained.
- Possible improvements: feature scaling, checking for outliers, and trying other models such as Ridge, Lasso, Random Forest or Gradient Boosting to compare performance.

## Author

Kinnery Anyanga
