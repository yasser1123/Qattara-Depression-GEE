# Qattara Depression GEE Project 🌍

A Google Earth Engine (GEE) and machine learning-based project for monitoring environmental changes, analyzing weather patterns, and visualizing satellite imagery for the **Qattara Depression** region.

## Features ✨
- **Satellite Image Processing**: Fetches and processes Landsat/Sentinel-2 imagery.
- **Time-Series Analysis**: Tracks environmental changes over time.
- **Vegetation & Water Index Calculation**: Computes NDVI and NDWI for vegetation and water body monitoring.
- **Multi-Threaded Data Fetching**: Collects weather data efficiently using parallel processing.
- **Machine Learning Predictions**: Uses Random Forest for weather forecasting.
- **Interactive Visualization**: Generates maps and Excel reports.
- **Customizable Parameters**: Select specific time ranges and regions of interest.

## Tech Stack 🛠️
- **Google Earth Engine (GEE)**: Cloud-based geospatial processing.
- **Python**: API interaction, data processing, and ML modeling.
- **JavaScript**: GEE scripts for interactive visualization.
- **Jupyter Notebook**: Running and analyzing results.
- **Threading (Python)**: Multi-threaded data fetching and processing.
- **scikit-learn**: Machine learning-based weather predictions.

## Project Structure 📂
```
Qattara-GEE-Project/
│── main.py                   # Main script to run the project
│── threads.py                # Multi-threaded data fetching and processing
│── Map-Tiff.py               # /Generates Tiff maps
│── weatherRelations.py       # Weather data aggregation and Excel generation
│── model.py                  # Machine learning model for weather forecasting
│── utils.py                  # All utilities used in the main file
│── requirements.txt          # Required dependencies
```

## Installation 🚀
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/Qattara-GEE-Project.git
   cd Qattara-GEE-Project
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Authenticate with Google Earth Engine:
   ```sh
   earthengine authenticate
   ```
4. Run the Python script:
   ```sh
   python main.py
   ```


## How It Works ⚙️

### Multi-Threaded Data Processing
- **DataFetcher**: Fetches weather data from GEE.
- **DataOrganizer**: Cleans and processes raw data.
- **WeatherAggregator**: Aggregates statistics and generates reports.
- **RealWeatherWriter**: Saves historical weather data.
- **Machine Learning Model**: Predicts precipitation using **Random Forest Regressor**.

### Satellite Data Processing
- Fetches Landsat/Sentinel-2 imagery.
- Computes **NDVI** (Normalized Difference Vegetation Index).
- Computes **NDWI** (Normalized Difference Water Index).
- Saves outputs as **GeoTIFF** files for GIS analysis.

### Machine Learning for Weather Forecasting
- Uses **Random Forest** for predicting precipitation.
- Extracts **time-series features** for modeling.
- Evaluates the model with **R² Score, MSE, MAE**.
- Saves predictions in `predicted_weather.csv`.


## Contribution Guide 🤝
1. **Fork the repository** and create a new branch:
   ```sh
   git checkout -b feature/new-feature
   ```
2. **Commit your changes**:
   ```sh
   git commit -m "Add new feature"
   ```
3. **Push to your branch**:
   ```sh
   git push origin feature/new-feature
   ```
4. **Open a pull request** and describe your updates!

## Future Enhancements 🚧
- Integrate **Deep Learning** for improved weather predictions.
- Expand to additional environmental indices (e.g., **LST - Land Surface Temperature**).
- Build a **web-based dashboard** for interactive data exploration.

---
🌍 **Qattara Depression GEE Project** - Empowering Environmental Insights!

