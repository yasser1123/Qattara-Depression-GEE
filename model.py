import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import queue


# Function to load 'weather data' sheet from Google Sheets
def load_google_sheet(sheet_url):
    sheet_id = sheet_url.split("/d/")[1].split("/")[0]
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet=weather%20data"
    return pd.read_csv(csv_url)

# Feature Engineering
def preprocess_data(data):
    # Convert 'date' to datetime
    data['date'] = pd.to_datetime(data['date'])

    # Extract cyclical features from 'month'
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

    # Extract day of the week from 'date' (0 = Monday, 6 = Sunday)
    data['day_of_week'] = data['date'].dt.dayofweek

    # Lagged features for temperature and precipitation
    for lag in [1, 3, 7, 14, 30, 60]:  # Added more lags (14, 30-day lags)
        data[f'temperature_lag{lag}'] = data['temperature'].shift(lag)
        data[f'precipitation_lag{lag}'] = data['total_precipitation'].shift(lag)

    # Rolling statistics (7-day and 30-day windows)
    data['temp_rolling_mean_7'] = data['temperature'].rolling(window=7).mean()
    data['temp_rolling_std_7'] = data['temperature'].rolling(window=7).std()
    data['precip_rolling_mean_7'] = data['total_precipitation'].rolling(window=7).mean()
    data['precip_rolling_std_7'] = data['total_precipitation'].rolling(window=7).std()
    data['temp_rolling_mean_30'] = data['temperature'].rolling(window=30).mean()  # Added 30-day rolling mean
    data['temp_rolling_std_30'] = data['temperature'].rolling(window=30).std()  # Added 30-day rolling std
    data['precip_rolling_mean_30'] = data['total_precipitation'].rolling(window=30).mean()
    data['precip_rolling_std_30'] = data['total_precipitation'].rolling(window=30).std()

    # Interaction terms: Temperature * Humidity and others
    data['temp_humidity_interaction'] = data['temperature'] * data['relative_humidity']
   

    # Handle missing values from lag and rolling window calculations
    data = data.dropna()

    # Feature scaling: Standardize numeric columns
    scaler = StandardScaler()
    columns_to_scale = ['temperature', 'wind_speed', 'relative_humidity', 'specific_humidity', 
                        'soil_temperature', 'total_precipitation', 'temperature_lag1', 'temperature_lag3', 
                        'temperature_lag7', 'temperature_lag14', 'temperature_lag30', 'temperature_lag60', 'precipitation_lag1', 
                        'precipitation_lag3', 'precipitation_lag7', 'precipitation_lag14', 'precipitation_lag30', 'precipitation_lag60', 
                        'temp_rolling_mean_7', 'temp_rolling_std_7', 'precip_rolling_mean_7', 'precip_rolling_std_7',
                        'temp_rolling_mean_30', 'temp_rolling_std_30', 'month_sin', 'month_cos', 'day_of_week', 
                        'temp_humidity_interaction', 'precip_rolling_mean_30','precip_rolling_std_30']
  
    # Ensure the columns to scale are in float64 before scaling
    data.loc[:, columns_to_scale] = data.loc[:, columns_to_scale].astype('float64')
    
    # Apply scaling to selected columns
    data.loc[:, columns_to_scale] = scaler.fit_transform(data.loc[:, columns_to_scale])

    # Split features and target
    X = data.drop(columns=['total_precipitation', 'date', 'month'])  # Drop target and non-feature columns
    y = data['total_precipitation']
    
    return X, y




def train_and_evaluate(X, y):
    # Define cross-validation strategy (time-series cross-validation)
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Random Forest Regressor model
    rf = RandomForestRegressor(random_state=42)
    
    # Simplified hyperparameter grid for RandomizedSearchCV
    param_dist = {
        'n_estimators': [50, 100, 200],  # Fewer estimators to speed up search
        'max_depth': [5, 10, 20],         # Fewer depth options
        'min_samples_split': [2, 10],     # Fewer split options
    }
    
    # Reduce the number of iterations to speed up the search
    random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=20, cv=tscv, verbose=2, random_state=42, n_jobs=-1)
    
    print("Starting hyperparameter tuning...")
    random_search.fit(X, y)  # Fit the RandomizedSearchCV
    
    # Get the best model from RandomizedSearchCV
    best_rf_model = random_search.best_estimator_
    
    # Predictions
    y_pred = best_rf_model.predict(X)
    
    # Evaluate performance
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")
    
    # Feature importance (for RandomForest and other tree-based models)
    feature_importances = best_rf_model.feature_importances_
    feature_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    print("Feature Importance:")
    print(feature_df)
    
    # Combine X, y, and y_pred into one DataFrame
    result_df = X.copy()  # Start with the features X
    result_df['y'] = y    # Add the true target variable y
    result_df['y_pred'] = y_pred  # Add the predicted values y_pred
    
    return result_df, best_rf_model, r2

# Main function to run the pipeline
def main(sheet_url):
    print("Loading data from Google Sheets...")
    data = load_google_sheet(sheet_url)
    
    print("Preprocessing data...")
    X, y = preprocess_data(data)
    
    print("Training and evaluating models...")
    result_df, model, r2_score = train_and_evaluate(X, y)
    result_df = result_df.drop(['temperature_lag1', 'temperature_lag3', 
                        'temperature_lag7', 'temperature_lag14', 'temperature_lag30', 'temperature_lag60', 'precipitation_lag1', 
                        'precipitation_lag3', 'precipitation_lag7', 'precipitation_lag14', 'precipitation_lag30', 'precipitation_lag60', 
                        'temp_rolling_mean_7', 'temp_rolling_std_7', 'precip_rolling_mean_7', 'precip_rolling_std_7',
                        'temp_rolling_mean_30', 'temp_rolling_std_30', 'month_sin', 'month_cos', 'day_of_week', 
                        'temp_humidity_interaction', 'precip_rolling_mean_30','precip_rolling_std_30'])
    result_df.to_csv("predicted_weather.csv", index=False)
    
    # Call your function (assum
    
    print(f"Best model R2 Score: {r2_score}")
    if r2_score >= 0.9:
        print("Model achieved 90%+ accuracy!")
    else:
        print("Model did not achieve 90%+ accuracy. Consider further tuning or feature engineering.", r2_score)
    
    return model

# Run the pipeline with your Google Sheet URL
sheet_url = "https://docs.google.com/spreadsheets/d/1px8P_cAgwvVaJKZvf2935-6gUXWoEZk6Hg9H-_KO8gw/edit?usp=sharing"
main(sheet_url)


if __name__ == "__main__":
    sheet_url = "https://docs.google.com/spreadsheets/d/1px8P_cAgwvVaJKZvf2935-6gUXWoEZk6Hg9H-_KO8gw/edit?usp=sharing"
    main(sheet_url)

# Save the model to a .pkl file
# model_save_path = "Qattara_model.pkl"
# with open(model_save_path,'wb') as file:
#     pickle.dump(random_search,file)