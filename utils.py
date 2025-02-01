import numpy as np
import pandas as pd

def calculate_relative_humidity(temp_k, dewpoint_k):
    temp_c = temp_k - 273.15
    dewpoint_c = dewpoint_k - 273.15
    actual_vapor_pressure = 6.112 * np.exp((17.67 * dewpoint_c) / (dewpoint_c + 243.5))
    saturation_vapor_pressure = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    return np.clip(actual_vapor_pressure / saturation_vapor_pressure, 0, 1)

def calculate_specific_humidity(temp_k, dewpoint_k, pressure):
    pressure_hpa = pressure / 100
    dewpoint_c = dewpoint_k - 273.15
    actual_vapor_pressure = 6.112 * np.exp((17.67 * dewpoint_c) / (dewpoint_c + 243.5))
    q = (0.622 * actual_vapor_pressure) / ((pressure_hpa) - (0.378 * actual_vapor_pressure))
    return q * 1000

def calculate_stats_by_year_and_variable(df):
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    weather_variables = {
        'temperature': 'Air Temperature (°C)',
        'relative_humidity': 'Humidity (°C)',
        'wind_speed': 'Wind Speed (m/s)',
        'wind_direction': 'Wind Direction (°)',
        'total_precipitation': 'Precipitation (mm)',
        'soil_temperature': 'Soil Temperature (°C)',
        # 'evaporation': 'Evaporation (mm)'
    }

    stats = []
    for column, variable_name in weather_variables.items():
        grouped_stats = df.groupby(['year']).agg(
            Average=(column, 'mean'),
            Standard_Deviation=(column, 'std'),
            Minimum=(column, 'min'),
            Maximum=(column, 'max')
        ).reset_index()
        grouped_stats['Weather Variable'] = variable_name
        stats.append(grouped_stats)

    stats_df = pd.concat(stats, ignore_index=True)
    stats_df = stats_df[['year', 'Weather Variable', 'Average', 'Standard_Deviation', 'Minimum', 'Maximum']]

    return stats_df

