import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.ticker as ticker
import os


# Load and Clean Dataset

CHARTS_PATH = "charts"
os.makedirs(CHARTS_PATH, exist_ok=True)

df = pd.read_csv('male_players.csv', encoding='latin1')

required_columns = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'movement_reactions','movement_agility','mentality_aggression','mentality_vision','mentality_composure', 'age', 'value_eur']
df = df[required_columns].dropna()

df['value_million'] = df['value_eur'] / 1_000_000
df = df[df['value_million'] < 100]  # cap extreme outliers

#interaction terms and non linear terms 
df['age_squared'] = df['age']**2
df['pace*age'] = df['pace']*df['age']
df['dribbling_squared'] = df['dribbling'] ** 2 
df['pace_squared'] = df['pace'] ** 2          
df['passing_dribbling'] = df['passing'] * df['dribbling']



X_pre = df[['pace','shooting', 'passing', 'dribbling', 'defending', 'movement_reactions','movement_agility','mentality_aggression','mentality_vision','mentality_composure', 'age']]
X = df[['pace','passing_dribbling','pace_squared','dribbling_squared','shooting', 'passing', 'dribbling', 'defending', 'movement_reactions','movement_agility','mentality_aggression','mentality_vision','mentality_composure', 'age','age_squared','pace*age']]
y = np.log1p(df['value_million'])  # log transform target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=71)
X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(X_pre, y, test_size=0.3, random_state=71) 



# Multiple Linear Regression without polynomial features or Interaction features ()

lin_model = LinearRegression()
lin_model.fit(X_train_pre, y_train_pre)

y_pred_lin_log = lin_model.predict(X_test_pre)
y_pred_lin = np.expm1(y_pred_lin_log)
y_test_actual_pre = np.expm1(y_test_pre)

lin1_r2 = r2_score(y_test_actual_pre, y_pred_lin)
lin1_mse = mean_squared_error(y_test_actual_pre, y_pred_lin)
lin1_mae = mean_absolute_error(y_test_actual_pre,y_pred_lin)

# Multiple Linear Regression with interaction terms and polynomial features 

lin_model2 = LinearRegression()
lin_model2.fit(X_train, y_train)

y_pred_lin_log = lin_model2.predict(X_test)
y_pred_lin = np.expm1(y_pred_lin_log)
y_test_actual = np.expm1(y_test)

lin2_r2 = r2_score(y_test_actual, y_pred_lin)
lin2_mse = mean_squared_error(y_test_actual, y_pred_lin)
lin2_mae = mean_absolute_error(y_test_actual,y_pred_lin)



# Polynomial Regression (Degree 2)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_pre)
X_test_poly = poly.transform(X_test_pre)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train_pre)

y_pred_poly_log = poly_model.predict(X_test_poly)
y_pred_poly = np.expm1(y_pred_poly_log)

poly_r2 = r2_score(y_test_actual_pre, y_pred_poly)
poly_mse = mean_squared_error(y_test_actual_pre, y_pred_poly)
poly_mae =mean_absolute_error(y_test_actual_pre,y_pred_poly)


# Random Forest Regressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf_log = rf_model.predict(X_test)
y_pred_rf = np.expm1(y_pred_rf_log)

rf_r2 = r2_score(y_test_actual, y_pred_rf)
rf_mse = mean_squared_error(y_test_actual, y_pred_rf)
rf_mae = mean_absolute_error(y_test_actual,y_pred_rf)


#  Results

print("\nModel Performance:")
print(f"Linear Regression without interactive and polynomial features, R squared: {lin1_r2:.3f}, MSE: {lin1_mse:.3f}, MAE: {lin1_mae:.3f}")
print(f"Linear Regression with interactive and polynomial features, R squared  : {lin2_r2:.3f}, MSE: {lin2_mse:.3f}, MAE: {lin2_mae:.3f}")
print(f"Polynomial Regression, R squared: {poly_r2:.3f}, MSE: {poly_mse:.3f}, MAE: {poly_mae:.3f}")
print(f"Random Forest, R squared: {rf_r2:.3f}, MSE: {rf_mse:.3f}, MAE: {rf_mae:.3f}")


# Visualization: Predicted vs Actual (All Models)

plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test_actual, y=y_pred_lin, alpha=0.5, label="Linear")
sns.scatterplot(x=y_test_actual, y=y_pred_poly, alpha=0.5, label="Polynomial")
sns.scatterplot(x=y_test_actual, y=y_pred_rf, alpha=0.5, label="Random Forest")
plt.plot([y_test_actual.min(), y_test_actual.max()],
         [y_test_actual.min(), y_test_actual.max()], 'r--')

plt.xlabel("Actual Value (€M)")
plt.ylabel("Predicted Value (€M)")
plt.title("Predicted vs Actual Market Value (All Models)")
plt.grid(alpha=0.3)


plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(10))

plt.legend()
plt.savefig(os.path.join(CHARTS_PATH, "predicted_vs_actual_all_models.png"))
plt.close()



# Feature Importance


# Linear Regression Coefficients (With Interaction Features)
lin_coefficients = pd.Series(lin_model2.coef_, index=X.columns)
plt.figure(figsize=(8,5))
lin_coefficients.sort_values().plot(kind='barh', color='skyblue')
plt.title("Linear Regression Feature Coefficients")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_PATH, "linear_coefficients.png"))
plt.close()



# Model Performance Bar Chart

models = ["Linear (No Poly)", "Linear (With Poly)", "Polynomial", "Random Forest"]
r2_scores = [lin1_r2, lin2_r2, poly_r2, rf_r2]

plt.figure(figsize=(8,5))
sns.barplot(x=models, y=r2_scores, palette="viridis",legend=False, hue=models)
plt.ylabel("R² Score")
plt.ylim(0,1)
plt.title("Model Performance Comparison (R²)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_PATH, "model_performance_r2.png"))
plt.close()


# Calculate correlation matrix
correlation_matrix = X.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, 
            annot=True,      
            cmap='coolwarm', 
            center=0,        
            square=True,     
            fmt='.2f')       
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_PATH,"Correlation_Matrix"))
plt.close()
