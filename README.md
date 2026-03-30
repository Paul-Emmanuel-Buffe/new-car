# Used Car Price Prediction: Linear Regression Analysis

## Project Overview
This project is developed within the framework of data expertise applied to the automotive market. The primary objective is to leverage machine learning to understand the factors influencing the valuation of used vehicles. By analyzing data from the CarDekho platform, this tool provides an objective estimation of a car's fair market value to assist buyers in informed decision-making.

## Dataset Specifications
The analysis is conducted on a dataset containing detailed characteristics of various vehicles. The features include:
* **Target Variable**: `Selling_Price` (the price at which the owner wishes to sell).
* **Predictors**: Year of manufacture, factory price (`Present_Price`), distance traveled (`Kms_Driven`), fuel type, seller type (dealer or individual), transmission mode, and number of previous owners.

## Technical Context: Linear Regression
The core of this study relies on **Linear Regression**, a machine learning method used to discover and quantify the relationship between independent variables (such as age and mileage) and a continuous target variable (price). The project implements both univariate and multivariate approaches to refine prediction accuracy.

## Technical Workflow
The development follows a structured data science pipeline:
* **Data Ingestion & Quality Audit**: Data loading via **Pandas**. Removal of duplicates to ensure statistical impartiality.
* **Exploratory Data Analysis (EDA)**: 
    * Univariate analysis using histograms and Kernel Density Estimation (KDE) to identify price distribution patterns, revealing a **positive asymmetry (Right Skew)**.
    * Outlier detection using the **IQR (Interquartile Range)** method to identify and filter atypical entries.
    * Correlation analysis using the **Pearson coefficient** to quantify the strength of linear relationships between predictors.
* **Feature Engineering**: 
    * **Encoding**: Mapping binary features and implementing **One-Hot Encoding** for categorical variables to avoid artificial ordering.
    * **Standardization**: Application of `StandardScaler` to normalize numerical features for optimal model convergence.
* **Modeling & Evaluation**: 
    * Benchmark univariate regression vs. multivariate regression.
    * Performance validation using **Stratified 5-Fold Cross-Validation** to ensure robustness across different price segments.

## Model Performance
The final multivariate model demonstrates high predictive power:
* **R² (Coefficient of Determination)**: 0.87 (explaining 87% of the variance).
* **MAE (Mean Absolute Error)**: 1.13 units.
* **RMSE (Root Mean Squared Error)**: 1.60 units.

## Case Study: Practical Estimation
To validate the model's utility, an estimation was performed for a specific request: a manual vehicle less than 7 years old with under 100,000 km.
* **Results**: For this specific segment, the model predicts an average price of **4.35 units**, with an R² of 0.89 on this subset, confirming high reliability for standard market requests.

## Repository Structure
* `new-car.ipynb`: Comprehensive notebook including EDA, data cleaning, and modeling.
* `linearegression.py`: (Bonus) Custom Linear Regression class implemented manually using **Numpy**.
* `/data`: Source dataset (carData.csv).