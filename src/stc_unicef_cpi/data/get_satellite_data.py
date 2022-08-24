from typing import Dict
import ee
import pycountry

import stc_unicef_cpi.utils.constants as c


class SatelliteImages:
    """Get Satellite Images From Google Earth Engine"""

    def __init__(
        self, country, folder=c.folder_ee, res=c.res_ee, start=c.start_ee, end=c.end_ee
    ):
        """Initialize class
        :param country: country
        :type country: str
        :param folder: folder path
        :type folder: str
        :param res: grid resolution
        :type res: int
        :param start: starting date
        :type start: str
        :param end: ending date
        :type end: str
        """
        country_record = pycountry.countries.search_fuzzy(country)[0]
        self.country = country_record.name
        self.country_code = country_record.alpha_3
        self.folder = folder + "/" + self.country
        self.res = res
        self.start = start
        self.end = end
        self.get_satellite_images()

    def get_country_boundaries(self):
        """Get countries boundaries"""
        countries = ee.FeatureCollection("FAO/GAUL/2015/level0").select("ADM0_NAME")
        ctry = countries.filter(ee.Filter.eq("ADM0_NAME", self.country))
        geo = ctry.geometry()

        return ctry, geo

    def get_projection(self):
        """Get country's transform between projected coordinates and the base coordinate system
        :return: the transform, the base coordinate reference system
        :rtype: List, Object
        """
        pop_tot = ee.Image(
            f"WorldPop/GP/100m/pop_age_sex/{self.country_code.upper()}_2020"
        )
        proj = pop_tot.select("population").projection().getInfo()

        return proj["transform"], proj["crs"]

    def task_config(self, geo, name, image, transform, proj) -> dict:
        """Determine countries parameters"""
        config = {
            "region": geo,
            "description": f"{name}_{self.country.lower()}_{self.res}",
            "crs": proj,
            "crsTransform": transform,
            "fileFormat": "GeoTIFF",
            "folder": self.folder,
            "scale": self.res,
            "image": image,
            "maxPixels": 9e8,
        }
        return config

    def export_drive(self, config) -> dict:
        """Export tiff file into drive
        :param config: Configuration of output tiff
        :type config: dictionary
        """
        task = ee.batch.Export.image.toDrive(**config)
        task.start()

        return task

    def get_pop_data(self, transform, proj, geo, name="cpi_poptotal") -> dict:
        """Get 2020 population estimates in country, by age and sex
        :param transform: transform between projected coordinates and the base coordinate system
        :type transform: list
        :param proj: the base coordinate reference system
        :type proj: object
        :param geo: dissolved geometry of all features in the collection
        :type geo: geometry
        :param name: name of file, defaults to "cpi_poptotal"
        :type name: str, optional
        :return: task status
        :rtype: dictionary
        """
        ctry, geo = self.get_country_boundaries()
        transform, proj = self.get_projection()
        pop_tot = ee.Image(
            f"WorldPop/GP/100m/pop_age_sex/{self.country_code.upper()}_2020"
        ).clip(ctry)
        config = self.task_config(geo, name, pop_tot, transform, proj)
        task = self.export_drive(config)

        return task

    def get_precipitation_data(
        self, transform, proj, ctry, geo, start_date, end_date
    ) -> dict:
        """Get precipitation data
        :param transform: transform between projected coordinates and the base coordinate system
        :type transform: list
        :param proj: the base coordinate reference system
        :type proj: object
        :param ctry: country of interest
        :type ctry: str
        :param geo: dissolved geometry of all features in the collection
        :type geo: geometry
        :param start_date: starting date
        :type start_date: str
        :param end_date: ending date
        :type end_date: str
        :return: task status
        :rtype: dictionary
        """
        # nasa data
        precip = (
            ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V06")
            .select("precipitation")
            .filterBounds(ctry)
            .filterDate(start_date, end_date)
            .select("precipitation")
        )
        precip_mean = precip.reduce(ee.Reducer.mean()).resample().clip(ctry)
        precip_std = precip.reduce(ee.Reducer.stdDev()).resample().clip(ctry)

        # idaho data
        terra_climate = (
            ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
            .filterBounds(ctry)
            .filterDate(start_date, end_date)
        )
        climate_evap = (
            terra_climate.select("aet").reduce(ee.Reducer.mean()).resample().clip(ctry)
        )
        climate_preci = (
            terra_climate.select("pr").reduce(ee.Reducer.mean()).resample().clip(ctry)
        )
        latest_image = terra_climate.limit(1, "system:time_start", False).first()
        drought = latest_image.select("pdsi").resample().clip(ctry)

        # collect data
        images = [precip_std, precip_mean, climate_preci, climate_evap, drought]
        names = [
            "cpi_preci_mean",
            "cpi_preci_std",
            "cpi_precipi_acc",
            "cpi_evapo_trans",
            "cpi_pdsi",
        ]
        collections = zip(images, names)
        for image, name in collections:
            config = self.task_config(geo, name, image, transform, proj)
            task = self.export_drive(config)
        return task

    def get_copernicus_data(
        self, transform, proj, ctry, geo, start_date, end_date, name="cpi_cop_land"
    ) -> dict:
        """Get status and evolution of land surface at global scale
        :param transform: transform between projected coordinates and the base coordinate system
        :type transform: list
        :param proj: the base coordinate reference system
        :type proj: object
        :param ctry: country of interest
        :type ctry: str
        :param geo: dissolved geometry of all features in the collection
        :type geo: geometry
        :param start_date: starting date
        :type start_date: str
        :param end_date: ending date
        :type end_date: str
        :param name: name of file, defaults to "cpi_cop_land"
        :type name: str, optional
        :return: task status
        :rtype: dictionary
        """
        cop_land_use = (
            ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V-C3/Global")
            .select("discrete_classification")
            .filterDate(start_date, end_date)
            .filterBounds(ctry)
        )
        cop_land_use = cop_land_use.reduce(ee.Reducer.mean()).clip(ctry)
        config = self.task_config(geo, name, cop_land_use, transform, proj)
        task = self.export_drive(config)

        return task

    def get_land_use_data(
        self, transform, proj, ctry, geo, name="cpi_ghsl"
    ) -> dict:
        """Get land use data
        :param transform: transform between projected coordinates and the base coordinate system
        :type transform: list
        :param proj: the base coordinate reference system
        :type proj: object
        :param ctry: country of interest
        :type ctry: str
        :param geo: dissolved geometry of all features in the collection
        :type geo: geometry
        :param name: name of file, defaults to "cpi_ghsl"
        :type name: str, optional
        :return: task status
        :rtype: dictionary
        """
        ghsl_land_use = (
            ee.Image("JRC/GHSL/P2016/BUILT_LDSMT_GLOBE_V1")
            .select("built", "cnfd")
            .clip(ctry)
        )
        config = self.task_config(geo, name, ghsl_land_use, transform, proj)
        task = self.export_drive(config)

        return task

    def get_ndwi_data(
        self, transform, proj, ctry, geo, start_date, end_date, name="cpi_ndwi"
    ) -> dict:
        """Get Normalized Difference Water Index (NDWI)
        :param transform: transform between projected coordinates and the base coordinate system
        :type transform: list
        :param proj: the base coordinate reference system
        :type proj: object
        :param ctry: country of interest
        :type ctry: str
        :param geo: dissolved geometry of all features in the collection
        :type geo: geometry
        :param start_date: starting date
        :type start_date: str
        :param end_date: ending date
        :type end_date: str
        :param name: name of file, defaults to "cpi_ndwi"
        :type name: str, optional
        :return: task status
        :rtype: dictionary
        """
        ndwi = (
            ee.ImageCollection("LANDSAT/LC08/C01/T1_ANNUAL_NDWI")
            .filterDate(start_date, end_date)
            .filterBounds(ctry)
            .select("NDWI")
        )
        ndwi = ndwi.reduce(ee.Reducer.mean()).clip(ctry)
        config = self.task_config(geo, name, ndwi, transform, proj)
        task = self.export_drive(config)
        return task

    def get_ndvi_data(
        self, transform, proj, ctry, geo, start_date, end_date, name="cpi_ndvi"
    ) -> dict:
        """Get Normalized Difference Vegetation Index 
        :param transform: transform between projected coordinates and the base coordinate system
        :type transform: list
        :param proj: the base coordinate reference system
        :type proj: object
        :param ctry: country of interest
        :type ctry: str
        :param geo: dissolved geometry of all features in the collection
        :type geo: geometry
        :param start_date: starting date
        :type start_date: str
        :param end_date: ending date
        :type end_date: str
        :param name: name of file, defaults to "cpi_ndwi"
        :type name: str, optional
        :return: task status
        :rtype: dictionary
        """
        ndvi = (
            ee.ImageCollection("LANDSAT/LC08/C01/T1_32DAY_NDVI")
            .select("NDVI")
            .filterDate(start_date, end_date)
            .filterBounds(ctry)
        )
        ndvi = ndvi.reduce(ee.Reducer.mean()).clip(ctry)
        config = self.task_config(geo, name, ndvi, transform, proj)
        task = self.export_drive(config)
        return task

    def get_pollution_data(
        self, transform, proj, ctry, geo, start_date, end_date, name="cpi_pollution"
    ) -> dict:
        """Get pollution data
        :param transform: transform between projected coordinates and the base coordinate system
        :type transform: list
        :param proj: the base coordinate reference system
        :type proj: object
        :param ctry: country of interest
        :type ctry: str
        :param geo: dissolved geometry of all features in the collection
        :type geo: geometry
        :param start_date: starting date
        :type start_date: str
        :param end_date: ending date
        :type end_date: str
        :param name: name of file, defaults to "cpi_ndwi"
        :type name: str, optional
        :return: task status
        :rtype: dictionary
        """
        def func_pio(m):
            collection = (
                ee.ImageCollection("MODIS/006/MCD19A2_GRANULES")
                .filterDate(start_date, end_date)
                .filter(ee.Filter.calendarRange(m, m, "month"))
                .filterBounds(ctry)
                .select("Optical_Depth_047", "Optical_Depth_055")
                .mean()
                .set("month", m)
            )
            return collection

        months = ee.List.sequence(1, 12)
        pollution = ee.ImageCollection.fromImages(months.map(func_pio))
        pollution = pollution.mean().clip(ctry)
        config = self.task_config(geo, name, pollution, transform, proj)
        task = self.export_drive(config)
        return task

    def get_topography_data(self, transform, proj, ctry, geo):
        """Get topography data
        :param transform: transform between projected coordinates and the base coordinate system
        :type transform: list
        :param proj: the base coordinate reference system
        :type proj: object
        :param ctry: country of interest
        :type ctry: str
        :param geo: dissolved geometry of all features in the collection
        :type geo: geometry
        :return: task status
        :rtype: dictionary
        """
        elevation = ee.Image("CGIAR/SRTM90_V4").select("elevation").clip(ctry)
        slope = ee.Terrain.slope(elevation)
        images, names = [elevation, slope], ["cpi_elevation", "cpi_slope"]
        collections = zip(images, names)
        for image, name in collections:
            config = self.task_config(geo, name, image, transform, proj)
            task = self.export_drive(config)
        return task

    def get_nighttime_data(
        self, transform, proj, ctry, geo, start_date, end_date, name="cpi_nighttime"
    ) -> dict:
        """Get nighttime data
        :param transform: transform between projected coordinates and the base coordinate system
        :type transform: list
        :param proj: the base coordinate reference system
        :type proj: object
        :param ctry: country of interest
        :type ctry: str
        :param geo: dissolved geometry of all features in the collection
        :type geo: geometry
        :param start_date: starting date
        :type start_date: str
        :param end_date: ending date
        :type end_date: str
        :param name: name of file, defaults to "cpi_ndwi"
        :type name: str, optional
        :return: task status
        :rtype: dictionary
        """
        night_time = (
            ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
            .filterDate(start_date, end_date)
            .filterBounds(ctry)
            .select("avg_rad", "cf_cvg")
        )
        night_time = night_time.mean().toFloat().clip(ctry)
        config = self.task_config(geo, name, night_time, transform, proj)
        task = self.export_drive(config)
        return task

    def get_healthcare_data(
        self, transform, proj, ctry, geo, name="cpi_health_acc"
    ) -> dict:
        """Get health care data
        :param transform: transform between projected coordinates and the base coordinate system
        :type transform: list
        :param proj: the base coordinate reference system
        :type proj: object
        :param ctry: country of interest
        :type ctry: str
        :param geo: dissolved geometry of all features in the collection
        :type geo: geometry
        :param name: name of file, defaults to "cpi_health_acc"
        :type name: str, optional
        :return: task status
        :rtype: dictionary
        """
        health_acc = ee.Image("Oxford/MAP/accessibility_to_healthcare_2019").clip(ctry)
        config = self.task_config(geo, name, health_acc, transform, proj)
        task = self.export_drive(config)
        return task

    def get_satellite_images(self) -> None:
        """Get satellite images"""
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
        self.get_nighttime_data(transform, proj, ctry, geo, start_date, end_date)
        self.get_healthcare_data(transform, proj, ctry, geo)
