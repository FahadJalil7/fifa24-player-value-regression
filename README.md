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

