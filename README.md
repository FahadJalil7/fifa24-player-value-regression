# FIFA Player Value Prediction Regression Model Comparison

## Overview 📌
This project predicts **FIFA male players’ market value** (in millions of euros) using multiple regression techniques.  
We compare:
- **Linear Regression (No Polynomial Features)**
- **Linear Regression (With Interaction & Polynomial Features)**
- **Polynomial Regression (Degree 2)**
- **Random Forest Regression**

The goal is to analyze which attributes most influence a player’s value, evaluate and compare model performance, and visualize the results.

---

## 📊 Dataset
- **Source:** [Kaggle — FC 24 Male Players Dataset](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset/data?select=male_players.csv)
- **Features Used:**
  - Pace, Shooting, Passing, Dribbling, Defending
  - Movement Reactions, Movement Agility
  - Mentality Aggression, Vision, Composure
  - Age 
  - Derived features like Age², Pace×Age, Passing×Dribbling

- **Target Variable:**
  - `value_eur` (converted to millions, capped at €100M to remove outliers)

---

## Correlation Matrix Insights
The correlation matrix highlights several key relationships:
Movement_Reactions shows a strong positive correlation with Value — players with higher reactions tend to have higher market value.
Age has a moderate negative correlation with Value, but the relationship is non-linear (younger players tend to have more potential value, while older star players remain valuable).
Passing, Dribbling, and Composure are moderately correlated with Value and also correlated with each other, explaining why some linear regression coefficients appear counterintuitive.

---
## Coefficient Interpretation
Due to the inclusion of interaction and polynomial terms, individual coefficients in the linear regression models are less directly interpretable.
This is a common tradeoff in applied regression: increasing model complexity to improve predictive performance often reduces the clarity of individual coefficient values.The focus of this project is on predictive accuracy and model comparison rather than precise coefficient interpretation.

