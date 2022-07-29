# -*- coding: utf-8 -*-
import ee


class SatelliteImages():
    """Get Satellite Images From Google Earth Engine"""

    def __init__(self, country, folder, res):

        self.country = country
        self.folder = folder
        self.res = res
        self.start = '2010-01-01'
        self.end = '2020-01-01'

    def get_country_boundaries(self):
        """Get countries boundaries"""

        countries = ee.FeatureCollection("FAO/GAUL/2015/level0").select("ADM0_NAME")
        ctry = countries.filter(ee.Filter.eq("ADM0_NAME", self.country))
        geo = ctry.geometry()

        return ctry, geo

    def get_projection(self):
        """Get country projections for image download"""

        pop_tot = ee.Image('WorldPop/GP/100m/pop_age_sex/NGA_2020')
        proj = pop_tot.select('population').projection().getInfo()

        return proj['transform'], proj['crs']

    def task_config(self, geo, name, image, transform, proj):

        config = {
            'region': geo,
            'description': f"{name}_{self.country.lower()}_{self.res}",
            'crs': proj,
            'crsTransform': transform,
            'fileFormat': 'GeoTIFF',
            'folder': self.folder,
            'scale': self.res,
            'image': image,
            'maxPixels': 9e8
        }

        return config

    def run_task_ee(self, config):
        task = ee.batch.Export.image.toDrive(**config)
        task.start()

        return task

    def get_pop_data(self, transform, proj, geo, name='cpi_poptotal'):
        ctry, geo = self.get_country_boundaries()
        transform, proj = self.get_projection()
        pop_tot = ee.Image('WorldPop/GP/100m/pop_age_sex/NGA_2020').clip(ctry)
        config = self.task_config(geo, name, pop_tot, transform, proj)
        task = self.run_task_ee(config)
        
        return task

    def get_precipitation_data(self, transform, proj, ctry, geo, start_date, end_date):

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
            config = self.task_config(geo, name, image, transform, proj)
            task = self.run_task_ee(config)
        return task

    def get_copernicus_data(self, transform, proj, ctry, geo, start_date, end_date, name='cpi_cop_land'):
        cop_land_use = ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V-C3/Global").\
            select('discrete_classification','discrete_classification-proba').\
                filterBounds(ctry).filterDate(start_date, end_date)
        config = self.task_config(geo, name, cop_land_use, transform, proj)
        task = self.run_task_ee(config)

        return task

    def get_land_use_data(self, transform, proj, ctry, geo, name='cpi_ghsl'):
        ghsl_land_use = ee.Image("JRC/GHSL/P2016/BUILT_LDSMT_GLOBE_V1").select('built', 'cnfd').clip(ctry)
        config = self.task_config(geo, name, ghsl_land_use, transform, proj)
        task = self.run_task_ee(config)
        
        return task

    def get_ndwi_data(self, transform, proj, ctry, geo, start_date, end_date, name='cpi_ndwi'):
        ndwi = ee.ImageCollection('LANDSAT/LC08/C01/T1_ANNUAL_NDWI').\
            filterDate(start_date, end_date).\
                filterBounds(ctry).select('NDWI')
        ndwi = ndwi.reduce(ee.Reducer.mean()).clip(ctry)
        config = self.task_config(geo, name, ndwi, transform, proj)
        task = self.run_task_ee(config)
        return task

    def get_ndvi_data(self, transform, proj, ctry, geo, start_date, end_date, name='cpi_ndvi'):
        ndvi = ee.ImageCollection('LANDSAT/LC08/C01/T1_32DAY_NDVI').\
            select('NDVI').filterDate(start_date, end_date).filterBounds(ctry)
        ndvi = ndvi.reduce(ee.Reducer.mean()).clip(ctry)
        config = self.task_config(geo, name, ndvi, transform, proj)
        task = self.run_task_ee(config)  
        return task

    def get_pollution_data(self, transform, proj, ctry, geo, start_date, end_date, name='cpi_pollution'):
        
        months = ee.List.sequence(1, 12)
        def func_pio(m, ctry=ctry):
        
            collection = ee.ImageCollection('MODIS/006/MCD19A2_GRANULES').\
                filterDate(start_date, end_date).filter(ee.Filter.calendarRange(m, m, 'month')).\
                filterBounds(ctry).select('Optical_Depth_047', 'Optical_Depth_055').mean().\
                set('month', m)
            return collection
        pollution = ee.ImageCollection.fromImages(months.map(func_pio))
        pollution = pollution.mean().clip(ctry)
        config = self.task_config(geo, name, pollution, transform, proj)
        task = self.run_task_ee(config)
        return task

    def get_topography_data(self, transform, proj, ctry, geo, name='cpi_poptotal'):

        elevation = ee.Image('CGIAR/SRTM90_V4').select('elevation').clip(ctry)
        slope = ee.Terrain.slope(elevation)
        images, names = [elevation, slope], ['cpi_elevation', 'cpi_slope']
        collections = zip(images, names)
        for image, name in collections:
            config = self.task_config(geo, name, image, transform, proj)
            task = self.run_task_ee(config)
        return task

    def get_nightime_data(self, transform, proj, ctry, geo, start_date, end_date, name='cpi_poptotal'):
        night_time = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG').\
            filterDate(start_date, end_date).filterBounds(ctry).select('avg_rad', 'cf_cvg')
        night_time = night_time.mean().toFloat().clip(ctry)
        config = self.task_config(geo, name, night_time, transform, proj)
        task = self.run_task_ee(config)

        return task

    def get_satellite_images(self):
        """Get Satellite Images"""
        ee.Initialize()
        start_date, end_date = ee.Date(self.start), ee.Date(self.end)
        ctry, geo = self.get_country_boundaries()
        transform, proj = self.get_projection()
        self.get_pop_data(transform, proj, geo)
        self.get_precipitation_data(transform, proj, ctry, geo, start_date, end_date)
        self.get_copernicus_data(transform, proj, ctry, geo, start_date, end_date)
        self.get_land_use_data(transform, proj, ctry, geo)
        self.get_ndwi_data(transform, proj, ctry, geo, start_date, end_date)
        self.get_ndvi_data(transform, proj, ctry, geo, start_date, end_date)
        self.get_pollution_data(transform, proj, ctry, geo, start_date, end_date)
        self.get_topography_data(transform, proj, ctry, geo)
        self.get_nightime_data(transform, proj, ctry, geo, start_date, end_date)


SatelliteImages(country='Nigeria', folder='gee', res=500).get_satellite_images()