# -*- coding: utf-8 -*-
import ee


def get_country_boundaries(country):
    countries = ee.FeatureCollection("FAO/GAUL/2015/level0").select("ADM0_NAME")
    ctry = countries.filter(ee.Filter.eq("ADM0_NAME", country))
    geo = ctry.geometry()

    return ctry, geo


def get_projection():

    pop_tot = ee.Image('WorldPop/GP/100m/pop_age_sex/NGA_2020')
    proj = pop_tot.select('population').projection().getInfo()

    return proj['transform'], proj['crs']


def task_config(country, name, res, image, transform, proj, folder_name):
    config = {
        'region': country,
        'description': f"{name}_{res}",
        'crs': proj,
        'crsTransform': transform,
        'fileFormat': 'GeoTIFF',
        'folder': folder_name,    
        'scale': res,
        'image': image,
        'maxPixels': 9e8
    }

    return config


def run_task_ee(config):
    task = ee.batch.Export.image.toDrive(**config)
    task.start()

    return task


def get_pop_data(country, res=500, name='cpi_poptotal'):
    ctry, geo = get_country_boundaries(country)
    transform, proj = get_projection(country)
    pop_tot = ee.Image('WorldPop/GP/100m/pop_age_sex/NGA_2020').clip(ctry)
    config = task_config(geo, name, res, pop_tot, transform, proj, 'gee')
    task = run_task_ee(config)
    
    return task


def get_precipitation_data(country, start='2010-01-01', end='2020-01-01', res=500):
    
    start_date, end_date = ee.Date(start), ee.Date(end)
    ctry, geo = get_country_boundaries(country)
    transform, proj = get_projection(country)

    # nasa data
    precip = ee.ImageCollection('NASA/GPM_L3/IMERG_MONTHLY_V06').\
        select('precipitation').filterBounds(ctry).filterDate(start_date, end_date).\
        select('precipitation')
    precip_mean = precip.reduce(ee.Reducer.mean()).resample().clip(ctry)
    precip_std = precip.reduce(ee.Reducer.stdDev()).resample().clip(ctry)
    
    # idaho data  
    terra_climate = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE').\
        filterBounds(ctry).filterDate(start_date, end_date)
    climate_evap = terra_climate.select('aet').reduce(ee.Reducer.mean()).resample().clip(ctry)
    climate_preci = terra_climate.select('pr').reduce(ee.Reducer.mean()).resample().clip(ctry)
    latest_image = terra_climate.limit(1, 'system:time_start', False).first()
    drought = latest_image.select('pdsi').resample().clip(ctry)
    
    # collect data
    images = [precip_std, precip_mean, climate_preci, climate_evap, drought]
    names = ['cpi_preci_mean', 'cpi_preci_std', 'cpi_precipi_acc', 'cpi_evapo_trans', 'cpi_pdsi']
    collections = zip(images, names)
    for image, name in collections:
        config = task_config(geo, name, res, image, transform, proj, 'gee')
        task = run_task_ee(config)
    return task


def get_copernicus_data(country, start='2010-01-01', end='2020-01-01', res=500, name='cpi_cop_land'):
    start_date, end_date = ee.Date(start), ee.Date(end)
    ctry, geo = get_country_boundaries(country)
    transform, proj = get_projection(country)
    cop_land_use = ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V-C3/Global").\
        select('discrete_classification','discrete_classification-proba').\
            filterBounds(ctry).filterDate(start_date, end_date);
    config = task_config(geo, name, res, cop_land_use, transform, proj, 'gee')
    task = run_task_ee(config)

    return task


def get_land_use_data(country, res=500, name='cpi_ghsl'):
    ctry, geo = get_country_boundaries(country)
    transform, proj = get_projection(country)
    ghsl_land_use = ee.Image("JRC/GHSL/P2016/BUILT_LDSMT_GLOBE_V1").select('built','cnfd').clip(ctry)
    config = task_config(geo, name, res, ghsl_land_use, transform, proj, 'gee')
    task = run_task_ee(config)
    
    return task


def get_ndwi_data(country, start='2010-01-01', end='2020-01-01', res=500, name='cpi_ndwi'):
    start_date, end_date = ee.Date(start), ee.Date(end)
    ctry, geo = get_country_boundaries(country)
    transform, proj = get_projection(country)
    ndwi = ee.ImageCollection('LANDSAT/LC08/C01/T1_ANNUAL_NDWI').\
        filterDate(start_date, end_date).\
            filterBounds(ctry).select('NDWI')
    ndwi = ndwi.reduce(ee.Reducer.mean()).clip(ctry)
    config = task_config(geo, name, res, ndwi, transform, proj, 'gee')
    task = run_task_ee(config)
    
    return task


def get_ndvi_data(country, start='2010-01-01', end='2020-01-01', res=500, name='cpi_ndvi'):
    start_date, end_date = ee.Date(start), ee.Date(end)
    ctry, geo = get_country_boundaries(country)
    transform, proj = get_projection(country)
    ndvi =  ee.ImageCollection('LANDSAT/LC08/C01/T1_32DAY_NDVI').\
    select('NDVI').filterDate(start_date, end_date).filterBounds(ctry)
    ndvi = ndvi.reduce(ee.Reducer.mean()).clip(ctry)
    config = task_config(geo, name, res, ndvi, transform, proj, 'gee')
    task = run_task_ee(config)
    
    return task


def get_pollution_data(country, res=500, name='cpi_pollution'):
    ctry, geo = get_country_boundaries(country)
    transform, proj = get_projection(country)
    months = ee.List.sequence(1, 12)
    
    def func_pio(m, ctry=ctry):
    
        collection = ee.ImageCollection('MODIS/006/MCD19A2_GRANULES').\
        filterDate(start_date, end_date).filter(ee.Filter.calendarRange(m, m, 'month')).\
        filterBounds(ctry).select('Optical_Depth_047', 'Optical_Depth_055').mean().set('month', m)
    
        return collection

    pollution = ee.ImageCollection.fromImages(months.map(func_pio))
    pollution = pollution.mean().clip(ctry)
    config = task_config(geo, name, res, pollution, transform, proj, 'gee')
    task = run_task_ee(config)
    
    return task



def get_topography_data(country, res=500):
    ctry, geo = get_country_boundaries(country)
    transform, proj = get_projection(country)
    elevation = ee.Image('CGIAR/SRTM90_V4').select('elevation').clip(ctry)
    slope = ee.Terrain.slope(elevation)
    # collect data
    images = [elevation, slope]
    names = ['cpi_elevation', 'cpi_slope']
    collections = zip(images, names)
    for image, name in collections:
        config = task_config(geo, name, res, image, transform, proj, 'gee')
        task = run_task_ee(config)
    return task


def get_nightime_data(country, start='2010-01-01', end='2020-01-01', res=500, name='cpi_poptotal'):
    start_date, end_date = ee.Date(start), ee.Date(end)
    ctry, geo = get_country_boundaries(country)
    transform, proj = get_projection(country)
    night_time = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG').\
        filterDate(start_date, end_date).filterBounds(ctry).select('avg_rad', 'cf_cvg')
    night_time = night_time.mean().toFloat().clip(ctry)
    config = task_config(geo, name, res, night_time, transform, proj, 'gee')
    task = run_task_ee(config)

    return task
