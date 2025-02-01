
point = ee.Geometry.Point([longitude, latitude])
region = point.buffer(200000).bounds()

landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA') \
    .filterDate(start_date, end_date) \
    .filterBounds(point)
# Function to Convert to tiff file format
def export_geotiff(image, filename, region, scale=30):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=filename,
        scale=scale,
        region=region,
        fileFormat='GeoTIFF'
    )
    task.start()
    return task


# Calculate NDVI for each image and get a composite
def add_ndvi(image):
    ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

ndvi_collection = landsat.map(add_ndvi)
ndvi_composite = ndvi_collection.select('NDVI').mean().clip(region)


dem = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(region)


# Converting NDVI and DEM to .tiff files' extension
ndvi_task = export_geotiff(ndvi_composite, 'NDVI', point)
        
dem_task = export_geotiff(dem, 'DEM', point)

# Visualization parameters
ndvi_vis = {
    'min': -1,
    'max': 1,
    'palette': ['blue', 'white', 'green']
}

dem_vis = {
    'min': 0,
    'max': 3000,
    'palette': ['white', 'gray', 'black']
}


# Folium map initialization
Map = geemap.Map(center=[latitude, longitude], zoom=10)

# Add layers
Map.addLayer(ndvi_composite, ndvi_vis, 'NDVI')
Map.addLayer(dem, dem_vis, 'DEM')

Map
