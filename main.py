import threading
from datetime import datetime, timedelta
from threads import DataFetcher, DataOrganizer, RealWeatherWriter
import queue
import ee
import pandas as pd
from weatherRelations import WeatherAggregator


ee.Authenticate()
ee.Initialize(project="qattara-depression-simulation")

threadLock = threading.Lock()

if __name__ == "__main__":
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 6)

    latitude = 29.9888
    longitude = 27.4997
    

    data_queue = queue.Queue()
    result_queue = queue.Queue()
    detailed_queue = queue.Queue()
    predicted_queue = queue.Queue()



    fetcher = DataFetcher(data_queue, start_date, end_date, latitude, longitude, threadLock)
    organizer = DataOrganizer(data_queue, result_queue, detailed_queue, predicted_queue, threadLock)
    realWriter = RealWeatherWriter(detailed_queue, threadLock)
    weather_relations = WeatherAggregator(detailed_queue, predicted_queue, threadLock)
    # predictWriter = PredictedWeatherWriter(predicted_queue, threadLock)
    # weatherCharts = WeatherChartWriter(detailed_queue, predicted_queue, threadLock)

    fetcher.start()
    organizer.start()
    realWriter.start()
    weather_relations.start()
    # predictWriter.start()
    # weatherCharts.start()

    fetcher.join()
    organizer.join()
    realWriter.join()
    weather_relations.join()
    # predictWriter.join()
    # weatherCharts.join()
