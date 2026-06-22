# Predictive Analytics

## Overview
Sales forecasting is critical for businesses to allocate resources, manage cash flow, and meet customer expectations. The BigMart Sales Prediction project explores data processing, exploratory data analysis, and the development of various machine-learning models to predict product sales in different stores.

---

## Aim
The goal of this project is to build and evaluate predictive models for sales forecasting, helping BigMart understand the factors influencing sales and develop better business strategies.

---

## Data Description
The dataset contains annual sales records for 1559 products across ten stores in different cities. Key attributes include:
- `item_identifier`: Unique item identifier
- `item_weight`: Item weight
- `item_fat_content`: Fat content in the item
- `item_visibility`: Product visibility in the outlet
- `item_type`: Product category
- `item_mrp`: Maximum retail price
- `outlet_identifier`: Outlet identifier
- `outlet_establishment_year`: Year of outlet establishment
- `outlet_size`: Outlet size
- `outlet_location_type`: Outlet location type
- `outlet_type`: Outlet type
- `item_outlet_sales`: Overall sales of the product in the outlet

---

## Tech Stack
- Language: `Python`
- Libraries: `Pandas`, `NumPy`, `Matplotlib`, `Scikit-learn`, `Redshift Connector`, `Pyearth`, `PyGAM`

---

## Approach
1. Data Exploration with Amazon Redshift
2. Data Cleaning and Imputation
3. Exploratory Data Analysis
   - Categorical Data
   - Continuous Data
   - Correlation
     - Pearson’s Correlation
     - Chi-squared Test and Contingency Tables
     - Cramer’s V Test
     - One-way ANOVA
4. Feature Engineering
   - Outlet Age
   - Label Encoding for Categorical Variables
5. Data Split
6. Model Building and Evaluation
   - Linear Regressor
   - Elastic Net Regressor
   - Random Forest Regressor
   - Extra Trees Regressor
   - Gradient Boosting Regressor
   - MLP Regressor
   - Multivariate Adaptive Regression Splines (MARS)
   - Spline Regressor
   - Generalized Additive Models
   - Voting Regressor
   - Stacking Regressor
   - Model Blending

---

## Code Structure
- `data`: Contains project data.
- `lib`: Reference notebooks.
- `ml_pipeline`: Python files for functions.
- `engine.py`: Main execution script.
- `requirements.txt`: List of required packages.
- `readme.md`: Instructions for running the code.

---

## Execution Instructions

- Create a python environment using the command 'python3 -m venv myenv'.
- Activate the environment by running the command 'myenv\Scripts\activate.bat'.
- Install the requirements using the command 'pip install -r requirements.txt'
- Run engine.py with the command 'python3 engine.py'.

---
