// NB JUST A TEMPLATE FOR TESTING
// Won't run alone - for asset generation in https://code.earthengine.google.com/

// Get a feature collection of administrative boundaries.
var countries = ee.FeatureCollection("FAO/GAUL/2015/level0").filterBounds(subSaharanAfrica).select("ADM0_NAME");

// Filter the feature collection to subset Nigeria.
var nigeria = countries.filter(ee.Filter.eq("ADM0_NAME", "Nigeria"));
print(nigeria);

var start_date = ee.Date('2010-01-01');
var end_date = ee.Date('2020-01-01');
var res_scale = 500;

///////////// POPULATION ////////////////
// Full dataset as follows
// var pop_data = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex")
//               .filterBounds(nigeria).select('population')
// NB pop_data has ~100m res

// Or specifically request Nigeria population data
// at specified scale and projection.
// var popMean = pop_data.toList(pop_data.size()).get(4);
var popTot = ee.Image('WorldPop/GP/100m/pop_age_sex/NGA_2020');
// .select(
//   'population','M_0','M_1','M_5','M_10','M_15','M_20','F_0','F_1','F_5','F_10','F_15','F_20'
//   );
popTot// Force the next reprojection to aggregate instead of resampling.
    // .reduceResolution({
    //   reducer: ee.Reducer.sum().unweighted(),
    //   maxPixels: 4096
    // })
    // // Request the data at the scale and projection of X image (if request projection as below).
    // .reproject({
    //   crs: 'EPSG:3857',
    //   scale: res_scale
    // })
    ;
print('popTot',popTot);

print('pop scale',popTot.select('population').projection().nominalScale())


Map.addLayer(popTot.clip(nigeria),
    // .map(function(img){return img.clip(nigeria)}),
    pop_vis, "Population");

//////////////// LAND USE /////////////
// COPERNICUS DATA
// See https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_Landcover_100m_Proba-V-C3_Global?hl=en&authuser=1
// for breakdown of classification value meaning
var copLandUse = ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V-C3/Global")
    .select('discrete_classification',
            'discrete_classification-proba'
    ).filterBounds(nigeria)
    .filterDate(start_date,end_date);
print('COPLAND',copLandUse);
// // Get information about the bands as a list.
// var copBandNames = copLandUse.mean().bandNames();
// print('COP band names:', copBandNames);  // ee.List of band names

Map.addLayer(copLandUse.select('discrete_classification').reduce(ee.Reducer.mean()).clip(nigeria),
    {}, "COP Land Use");

print('COP scale',copLandUse.select('discrete_classification').first().projection().nominalScale())
// GHSL DATA
// See https://developers.google.com/earth-engine/datasets/catalog/JRC_GHSL_P2016_BUILT_LDSMT_GLOBE_V1?hl=en&authuser=1
// for band details
// No temporal info directly - present directly in values
var ghslLandUse = ee.Image("JRC/GHSL/P2016/BUILT_LDSMT_GLOBE_V1")
      .select('built','cnfd')
      .clip(nigeria);
print('GHSL LAND',ghslLandUse);

print('GHSL scale',ghslLandUse.select('built').projection().nominalScale())

Map.addLayer(ghslLandUse,
    ghslVis, "GHSL");


/////////////////// NDWI ///////////////////
// See https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_ANNUAL_NDWI?hl=en&authuser=1#bands
// though only single band, between -1 and 1
// Supplementary vegetation index to NDVI, using different wavelengths
var ndwi = ee.ImageCollection('LANDSAT/LC08/C01/T1_ANNUAL_NDWI')
                  .filterDate(start_date,end_date)
                  .filterBounds(nigeria)
                  .select('NDWI');
print('NDWI',ndwi)
// NB seems like this is wrong - actual pixel size supposed to be ~30m
print('NDWI scale',ndwi.first().select('NDWI').projection().nominalScale())


Map.addLayer(ndwi.reduce(ee.Reducer.mean()).clip(nigeria), ndwiVis, 'NDWI');

//////////////// NDVI ///////////////
// See https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C01_T1_32DAY_NDVI?hl=en&authuser=1
// though again just single band NDVI between -1 and 1
var ndvi = ee.ImageCollection('LANDSAT/LC08/C01/T1_32DAY_NDVI')
                  .select('NDVI')
                  .filterDate(start_date,end_date)
                  .filterBounds(nigeria)
                  ;
print('NDVI',ndvi)
// NB seems like again this is wrong - actual pixel size supposed to be ~30m
print('NDVI scale',ndvi.first().select('NDVI').projection().nominalScale())


Map.addLayer(ndvi.reduce(ee.Reducer.mean()).clip(nigeria), ndviVis, 'NDVI');
// NB scale seems v low (~100km), but as composited image it ends up being reasonably fine
// var ndviScale = ndvi.reduce(ee.Reducer.mean()).projection().nominalScale();
// print('NDVI scale in meters', ndviScale);

/////////////// Topography (elevation + slope) ////////////////
// See https://developers.google.com/earth-engine/datasets/catalog/CGIAR_SRTM90_V4?hl=en&authuser=1#bands
// 90m res, single band (elevation)
var elevation = ee.Image('CGIAR/SRTM90_V4').select('elevation').clip(nigeria);

print('Elevation / slope scale',elevation.projection().nominalScale())

var slope = ee.Terrain.slope(elevation);
Map.addLayer(elevation,{min: 0, max: 2000},'elevation')
Map.addLayer(slope, {min: 0, max: 60}, 'slope');

/////// Pollution //////////
// See https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MCD19A2_GRANULES?hl=en&authuser=1
// 1km res
// Not 100% which bands to use, TODO check
// also 'Optical_Depth_055'
// NB data is daily, so significantly more than other examples - takes a little longer
// var startYear = 2010;
// var endYear = 2020;
// var years = ee.List.sequence(startYear, endYear);
var months = ee.List.sequence(1, 12);

// Group by month, and then reduce within groups by mean();
// the result is an ImageCollection with one image for each
// month.
var pollution = ee.ImageCollection.fromImages(
      months.map(function (m) {
        return ee.ImageCollection('MODIS/006/MCD19A2_GRANULES')
                  .filterDate(start_date,end_date)
                  .filter(ee.Filter.calendarRange(m,m,'month'))
                  .filterBounds(nigeria)
                  .select('Optical_Depth_047','Optical_Depth_055')
                  .mean()
                  .set('month', m);
                  }));


print('Pollution',pollution)
print('Pollution scale',pollution.select('Optical_Depth_047').first().projection().nominalScale())

Map.addLayer(pollution.select('Optical_Depth_047')
            .mean()
            .clip(nigeria),
            poll_band47_viz,
            'Pollution: Optical Depth 047');


///////// NIGHTTIME LIGHTS /////////////
// See https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMSLCFG?hl=en&authuser=1#bands
// ~460m res
// Two bands only - radiance and number of observations (lower means lower confidence)
// Monthly

var nighttime = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
                  .filterDate(start_date,end_date)
                  .filterBounds(nigeria)
                  .select('avg_rad','cf_cvg');

print('Nighttime',nighttime);
print('Nighttime scale',nighttime.select('avg_rad').first().projection().nominalScale())




Map.addLayer(nighttime.mean().clip(nigeria), nighttimeVis, 'Nighttime');


///////// PLOT //////////
// Map.setCenter(22.22,1.60, 6); // DRC
Map.setCenter(7.683939924577903,7.70381136085568,6); // Nigeria

// Fix the projection used to export, so all on same grid
var export_proj = popTot.select('population').projection().getInfo();

var fullCollection = ee.ImageCollection(
  [popTot.clip(nigeria),
  copLandUse
    // .select('discrete_classification')
    .reduce(ee.Reducer.mean())
    .clip(nigeria),
  ghslLandUse,
  ndwi.reduce(ee.Reducer.mean()).clip(nigeria),
  ndvi.reduce(ee.Reducer.mean()).clip(nigeria),
  elevation,
  slope,
  pollution.mean().clip(nigeria),
  nighttime.mean().clip(nigeria)]
  );
print('full',fullCollection)

// Can't export ImageCollection, only images, so do each separately
// First population
Export.image.toDrive({
  image: popTot.clip(nigeria),
  description: 'cpiPopData_'+res_scale,
  // assetId: 'cpiPopDatav1',
  fileFormat: 'GeoTIFF',
  region: nigeria,
  scale: res_scale,
  crs: export_proj.crs,
  crsTransform: export_proj.transform,
  folder: 'Data',
  maxPixels: 9e8
});

// Now Cop. Land
Export.image.toDrive({
  image: copLandUse
    // .select('discrete_classification')
    .reduce(ee.Reducer.mean())
    .clip(nigeria),
  description: 'cpiCopLandData_'+res_scale,
  // assetId: 'cpiCopLandDatav1',
  fileFormat: 'GeoTIFF',
  region: nigeria,
  scale: res_scale,
  crs: export_proj.crs,
  crsTransform: export_proj.transform,
  folder: 'Data',
  maxPixels: 9e8
});

// GHSL
Export.image.toDrive({
  image: ghslLandUse,
  description: 'cpiGHSLData_'+res_scale,
  // assetId: 'cpiGHSLDatav1',
  fileFormat: 'GeoTIFF',
  region: nigeria,
  scale: res_scale,
  crs: export_proj.crs,
  crsTransform: export_proj.transform,
  folder: 'Data',
  maxPixels: 9e8
});

// NDWI
Export.image.toDrive({
  image: ndwi.reduce(ee.Reducer.mean()).clip(nigeria),
  description: 'cpiNDWIData_'+res_scale,
  // assetId: 'cpiNDWIDatav1',
  fileFormat: 'GeoTIFF',
  region: nigeria,
  scale: res_scale,
  crs: export_proj.crs,
  crsTransform: export_proj.transform,
  folder: 'Data',
  maxPixels: 9e8
});

// NDVI
Export.image.toDrive({
  image: ndvi.reduce(ee.Reducer.mean()).clip(nigeria),
  description: 'cpiNDVIData_'+res_scale,
  // assetId: 'cpiNDVIDatav1',
  fileFormat: 'GeoTIFF',
  region: nigeria,
  scale: res_scale,
  crs: export_proj.crs,
  crsTransform: export_proj.transform,
  folder: 'Data',
  maxPixels: 9e8
});

// Elevation + slope
Export.image.toDrive({
  image: elevation,
  description: 'cpiElevationData_'+res_scale,
  // assetId: 'cpiElevationDatav1',
  fileFormat: 'GeoTIFF',
  region: nigeria,
  scale: res_scale,
  crs: export_proj.crs,
  crsTransform: export_proj.transform,
  folder: 'Data',
  maxPixels: 9e8
});
Export.image.toDrive({
  image: slope,
  description: 'cpiSlopeData_'+res_scale,
  // assetId: 'cpiSlopeDatav1',
  fileFormat: 'GeoTIFF',
  region: nigeria,
  scale: res_scale,
  crs: export_proj.crs,
  crsTransform: export_proj.transform,
  folder: 'Data',
  maxPixels: 9e8
});

// Pollution
Export.image.toDrive({
  image: pollution.mean().clip(nigeria),
  description: 'cpiPollutionData_'+res_scale,
  // assetId: 'cpiPollutionDatav1',
  fileFormat: 'GeoTIFF',
  region: nigeria,
  scale: res_scale,
  crs: export_proj.crs,
  crsTransform: export_proj.transform,
  folder: 'Data',
  maxPixels: 9e8
});


// Nighttime
Export.image.toDrive({
  image: nighttime.mean().toFloat().clip(nigeria),
  description: 'cpiNighttimeData_'+res_scale,
  // assetId: 'cpiNighttimeDatav1',
  fileFormat: 'GeoTIFF',
  region: nigeria,
  scale: res_scale,
  crs: export_proj.crs,
  crsTransform: export_proj.transform,
  folder: 'Data',
  maxPixels: 9e8
});

// CONSIDER ALT VEG  (enhanced version of NDVI)
// https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13A1

// CONSIDER INCLUDING S.D. values alongside mean (esp for e.g. pollution)
// Can combine as here
// https://developers.google.com/earth-engine/guides/reducers_intro




// Optionally retrieve the projection information from a band of the original image.
// Call getInfo() on the projection to request a client-side object containing
// the crs and transform information needed for the client-side Export function.
// var projection = landsat.select('B2').projection().getInfo();
// var projection = {crs: 'EPSG:3857',
//                   transform: {}};

// // Export the image to an Earth Engine asset.
// Export.image.toAsset({
//   image: fullCollection,
//   description: 'cpiSatData',
//   assetId: 'cpiSatDatav1',
//   // crs: projection.crs,
//   // crsTransform: projection.transform,
//   region: nigeria,
//   scale: 1000,
//   maxPixels: 9e8
//   // pyramidingPolicy: {
//   //   'b4_mean': 'mean',
//   //   'b4_sample': 'sample',
//   //   'b4_max': 'max'
//   // }
// });






// // Load an image.
// var landsat_img = ee.Image('LANDSAT/LC08/C01/T1/LC08_044034_20140318');

// // Define visualization parameters in an object literal.
// var landsat_vizParams = {bands: ['B5', 'B4', 'B3'], min: 5000, max:s 15000, gamma: 1.3};

// Center the map on the image and display.
// Map.centerObject(image, 9);
// Map.addLayer(landsat_img, landsat_vizParams, 'Landsat 8 false color');

// var landsat_collection = ee.ImageCollection('LANDSAT/LC08/C01/T1');

// var point = ee.Geometry.Point(22.22,1.60);

// var start = ee.Date('2014-06-01');
// var finish = ee.Date('2014-10-01');

// var filteredLandsatCollection = ee.ImageCollection('LANDSAT/LC08/C01/T1')
//   // .filterBounds(point)
//   .filterBounds(nigeria)
//   .filterDate(start, finish)
//   .sort('CLOUD_COVER');

// var first = filteredLandsatCollection.first();

// // Get information about the LANDSAT B5 projection.
// var landsatProjection = first.select('B5').projection();
// print('LANDSAT B5 projection:', landsatProjection);


// // This function gets NDVI from Landsat 8 imagery.
// var addNDVI = function(image) {
//   return image.addBands(image.normalizedDifference(['B5', 'B4']));
// };

// // Map the function over the collection.
// var ndviCollection = filteredLandsatCollection.map(addNDVI);

// // Compute the median of each pixel for each band of the 5 least cloudy scenes - make new image.
// // does worse than overall median
// var median = filteredLandsatCollection.limit(5).reduce(ee.Reducer.median()).reproject('EPSG:3857');

// // NB use print(...) to add items to console, which allows inspection of elements
// // This includes bands, with associated ranges, crs, and more
// print(median);

// var medvizParams = {bands: ['B4_median', 'B3_median', 'B2_median'], min: 0, max: 20000};
// Map.addLayer(median, medvizParams, 'median');


// Many different algorithms built in, e.g.
// Load the SRTM image.
// var srtm = ee.Image('CGIAR/SRTM90_V4');

// // Apply an algorithm to an image.
// var slope = ee.Terrain.slope(srtm);

// // Display the result.
// Map.setCenter(-112.8598, 36.2841, 9); // Center on the Grand Canyon.
// Map.addLayer(slope, {min: 0, max :60}, 'slope');

// print('Collection: ', filteredLandsatCollection);

// // Get the number of images.
// var count = filteredLandsatCollection.size();
// print('Count: ', count);

// // Get the date range of images in the collection.
// var range = filteredLandsatCollection.reduceColumns(ee.Reducer.minMax(), ['system:time_start']);
// print('Date range: ', ee.Date(range.get('min')), ee.Date(range.get('max')));

// // Get statistics for a property of the images in the collection.
// var sunStats = filteredLandsatCollection.aggregate_stats('SUN_ELEVATION');
// print('Sun elevation statistics: ', sunStats);

// // Get native scale of a band of an image as follows:
// var scale = first.select('B5').projection().nominalScale();
// print('Landsat B5 scale in meters', scale);

// If you specify a scale smaller than the native resolution,
// Earth Engine will happily resample the input image using
// nearest neighbor, then include all those smaller pixels in
// the computation. If you set the scale to be larger, Earth
// Engine will use input pixels from an aggregated version of
// the input (i.e. get pixels from a higher level of the
// image pyramid).

// // Sort by a cloud cover property, get the least cloudy image.
// var image = ee.Image(filteredLandsatCollection.sort('CLOUD_COVER').first());
// print('Least cloudy image: ', image);

// Also many mathematical ops, e.g.
// // Get the aspect (in degrees).
// var aspect = ee.Terrain.aspect(srtm);

// // Convert to radians, compute the sin of the aspect.
// var sinImage = aspect.divide(180).multiply(Math.PI).sin();

// // Display the result.
// Map.addLayer(sinImage, {min: -1, max: 1}, 'sin');
