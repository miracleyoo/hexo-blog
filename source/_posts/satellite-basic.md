---
title: Analyzing Geography Data (Beginners' Tutorial)
tags:
  - satellite
  - python
date: 2021-02-03 18:18:19
---

## 数据来源

### 卫星类型

1. Sentinel-2：提供混合分辨率的13 Bands MSI。分辨率有$10m \times 10m$，$20m \times 20m$，$60m \times 60m$，bands中心波长从442.3nm到2185.7nm。

   <img src="image-20201213154533413.png" alt="image-20201213154533413" style="zoom:33%;" />

   <!-- more -->

2. Planet：这个数据源提供$3m \times 3m$的4 bands MSI image。分别是R、G、B、NIR。

   - 这个数据源似乎只对美国国内的院校机构开放，不过不是很确定，需要的可以试试。

   ![Planet Specification](image-20201213150111581.png)

3. RapidEyes

### Sentinel-2 Data Source

1. USGS Earth Explorer: [Link](https://earthexplorer.usgs.gov/). It support downloading data within five years. And it not only support sentinel-2 data, but also contains many other satellite datasource.

   ![image-20201227143234319](image-20201227143234319.png)

2. Copernicus Open Access Hub: [Link](https://scihub.copernicus.eu/dhus/#/home). It supports only one year's old data. You can request the older data, but not guarante the fetch time.

   ![image-20201227145324164](image-20201227145324164.png)

3. Amazon AWS Sentinel-2 Service: [Link](https://registry.opendata.aws/sentinel-2/)

### Planet Data Source

- 官网：[Link](https://www.planet.com/)


### Specification

1. [Sentinel-2 MultiSpectral Instrument (MSI) Overview](https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/msi-instrument)
2. [Sentinel-2 Specification Doc](https://www.planet.com/products/satellite-imagery/files/Planet_Combined_Imagery_Product_Specs_December2017.pdf)
3. [Planet Specification](https://www.planet.com/products/planet-imagery/)

## 软件与Python包简介

- [GDAL](http://www.gdal.org/) –> Fundamental package for processing vector and raster data formats (many modules below depend on this). Used for raster processing.
- [Rasterio](https://github.com/mapbox/rasterio) –> Clean and fast and geospatial raster I/O for Python. [Guidebook](https://rasterio.readthedocs.io/en/latest/quickstart.html).
- [Geopandas](http://geopandas.org/#description) –> Working with geospatial data in Python made easier, combines the capabilities of pandas and shapely.
- [Shapely](http://toblerity.org/shapely/manual.html) –> Python package for manipulation and analysis of planar geometric objects (based on widely deployed [GEOS](https://trac.osgeo.org/geos/)).
- [Fiona](https://pypi.python.org/pypi/Fiona) –> Reading and writing spatial data (alternative for geopandas).
- [Pyproj](https://pypi.python.org/pypi/pyproj?) –> Performs cartographic transformations and geodetic computations (based on [PROJ.4](http://trac.osgeo.org/proj)).
- [Pysal](https://pysal.readthedocs.org/en/latest/) –> Library of spatial analysis functions written in Python.
- [Geopy](http://geopy.readthedocs.io/en/latest/) –> Geocoding library: coordinates to address <-> address to coordinates.
- [GeoViews](http://geo.holoviews.org/index.html) –> Interactive Maps for the web.
- [Networkx](https://networkx.github.io/documentation/networkx-1.10/overview.html) –> Network analysis and routing in Python (e.g. Dijkstra and A* -algorithms), see [this post](http://gis.stackexchange.com/questions/65056/is-it-possible-to-route-shapefiles-using-python-and-without-arcgis-qgis-or-pgr).
- [Cartopy](http://scitools.org.uk/cartopy/docs/latest/index.html) –> Make drawing maps for data analysis and visualisation as easy as possible.
- [Scipy.spatial](http://docs.scipy.org/doc/scipy/reference/spatial.html) –> Spatial algorithms and data structures.
- [Rtree](http://toblerity.org/rtree/) –> Spatial indexing for Python for quick spatial lookups.
- [RSGISLib](http://www.rsgislib.org/index.html#python-documentation) –> Remote Sensing and GIS Software Library for Python.
- [python-geojson](https://python-geojson.readthedocs.io/en/latest/)-> Deal with geojson format files.

## Sentinel-2 大气校正

Sentinel-2一般有两种standard，一个是A级别，一个是C级别。C级别的数据是你可以从任意网站上下载到的数据，它没有经过大气校正，是粗数据，每个区块的反射率可能有不同，不适合直接用作深度学习数据。

经过Sen2Cor软件校正后可以得到A级别的数据。

Sen2Cor是欧空局发布的一个软件，它即可以作为SNAP的插件安装，也可以作为独立命令行软件使用。推荐后者，更为快速、稳定，尤其适用于大量数据时，可以用脚本批处理。[官网链接](http://step.esa.int/main/snap-supported-plugins/sen2cor/sen2cor_v2-8/)

### 查询坐标映射

- 地球是圆的。
- 卫星上拍的一张矩形照片所对应的区域并非是矩形的。
- 使用卫星图片时要先将其映射到二维展开的坐标系中。
- 整个地球被分为了许多预先订好的区域。
- 每个区域有一个编号。
- 编号可以在[EPSG](http://epsg.io/)网站查询。

## SNAP

- 欧空局自己用作处理Sentinel-2的软件。[Link](https://step.esa.int/main/download/snap-download/)

- 只用来处理Sentinel-2，尤其是预处理，包括校正，reprojection，粗crop，统一各个bands分辨率等等，非常好用。因为是亲儿子，所以甚至可以直接读取Sentinel-2每个文件的压缩包，总之十分便利。

- 比较古老，interface有年代感，功能也相较于其他软件比较局限。

- 推荐用作第一步预处理。

- 使用流程：`读取->剪裁->correction->resize->reprojection->导出`。

- 大部分需要用到的功能都在`Raster->Geometric`中

- 经过尝试、搜索、确认，SNAP并不提供便利的选定区域截图，或是依据shapefile剪裁，所以目前最佳的方法是，先通过zoom地图和改变窗口大小确保需要的部分大致在view的可视范围内，然后右键，选择`Spatial Subset from View`，然后可视区域即可被剪裁。注意，这只是粗剪裁，所以尽量多包括一点，也不要少任何一部分。剪裁后的内容可以后续在ArcGIS Pro中进一步处理。

  <img src="image-20201227155911702.png" alt="image-20201227155911702" style="zoom:40%;" />

- 展示图：

  <img src="image-20201227155421761.png" alt="image-20201227155421761" style="zoom:33%;" />

  ![image-20201227155558109](image-20201227155558109.png)

## ArcGIS Pro

- 我能找到的最强大的可用于卫星图像处理的软件。

- 专业，美观，支持format多，有着强大的raster functions。

- 版权软件，下载之前先确认自己学校或公司是否提供License。

- 有时导出raster会出现随机bug，导致：导出可能是纯黑的图片，导出部分没有按照期望剪裁等。很恶心，但似乎也没有更好的选择。

- 解决导出问题：

  1. 关闭导出窗口，重新操作，多试几次。可以解决绝大部分问题。
  2. 和剪裁、mask等有关的问题可以先使用raster function进行这些操作，再直接导出前面操作的结果layer。

- 关于剪裁后的卫星图片和原始图片有着明显亮度对比度区别的问题：

  1. 首先，这不是一个bug，而是一个feature。。。实际上的卫星图都很暗沉的，所以软件原生提供一个“显示方法”的函数，调整图片曲线，使得显示的图片比较亮，容易看清楚细节。然而，剪裁后的图片有着不一样的统计值，所以在有些显示函数下，显示的结果和剪裁前结果不同。

     <img src="image-20201227155342029.png" alt="image-20201227155342029" style="zoom:33%;" />

  2. 但是不用担心，因为导出时候并不会考虑这个显示函数。即：剪裁前后的导出图像数值是相同的。

- 下面是两张展示图：

![image-20201227155051608](image-20201227155051608.png)

![image-20201227155259214](image-20201227155259214.png)

## QGIS

- 开源软件，支持很多插件，比如直接下载Sentinel-2数据，Sen2Cor等。[Link](https://qgis.org/en/site/)

- 支持很多format的数据。

- 问题是，不稳定，效率低，容易崩溃（软件&心态），甚至左侧Explorer遇到大文件夹都要经常转很久才能进去，或是干脆就直接转崩了。我怀疑他们要分析每个文件夹所有文件之后再显示列表。。。总之，慎重。

  ![image-20201227160344273](image-20201227160344273.png)

## Pythonic Method

虽然前面介绍了几个软件，但是说实话，处理一两个可以，批量处理几十几百甚至几十万就有点力不从心了。所以最后还是狠下心研究了一遍Python处理这些数据的方法，写出了一批适合我项目用的函数。不一定适合所有人，但可以作为参考：

```python
import numpy as np
import json
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import rasterio as rio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

def bbox(shp):
    """ Compute the bounding box of a certain shapefile.
    """
    piece = np.array([i.bounds for i in shp['geometry']])
    minx = piece[:,0].min()
    miny = piece[:,1].min()
    maxx = piece[:,2].max()
    maxy = piece[:,3].max()
    return minx, miny, maxx, maxy

def edge_length(shp):
    """ Compute the x and y edge length for a ceratin shapefile.
    """
    minx, miny, maxx, maxy = bbox(shp)
    return round(maxx-minx,3), round(maxy-miny,3)

def shape2latlong(shp):
    """ Turn the shapefile unit from meters/other units to lat/long.
    """
    return shp.to_crs(epsg=4326)

def bbox_latlong(shp):
    """ Compute the latitude-longitude bounding box of a certain shapefile.
    """
    shp = shape2latlong(shp)
    return bbox(shp)

def bbox_polygon(shp):
    """ Return the rectangular Polygon bounding box of a certain shapefile.
    """
    minx, miny, maxx, maxy = bbox(shp)
    return Polygon([(minx, miny), (minx, maxy), (maxx,maxy), (maxx, miny)])

def merge_polygon(shp):
    """ Merge a shapefile to one single polygon.
    """
    return shp.dissolve(by='Id').iloc[0].geometry

def polygon2geojson(polygon):
    """ Turn a polygon to a geojson format string.
        This is used for rasterio mask operation.
    """
    if type(polygon) == Polygon:
        polygon = gpd.GeoSeries(polygon)
    return [json.loads(polygon.to_json())['features'][0]['geometry']]

def sen2rgb(img, scale=30):
    """ Turn the 12 channel float32 format sentinel-2 images to a RGB uint8 image. 
    """
    return (img[(3,2,1),]/256*scale).astype(np.uint8)

def cropbyshp(raster, shp):
    """ Crop a raster using a shapefile.
    """
    # Reproject the shapefile to the same crs of raster.
    shp = shp.to_crs({"init": str(raster.crs)})
    # Compute the rectangular Polygon bounding box of a certain shapefile.
    bbpoly = bbox_polygon(shp)
    # Execute the mask operation.
    out_img, out_transform = mask(dataset=raster, shapes=polygon2geojson(bbpoly), crop=True, all_touched=True)
    return out_img

def write_raster(raster, path):
    """ Write a created raster object to file.
    """
    with rio.open(
        path,
        'w',
        **raster.meta
    ) as dst:
        dst.write(raster.read())

def sen_reproject(src, dst_crs, out_path):
    """ Reproject a raster to a new CRS coordinate, and save it in out_path.
    Args:
        src: Input raster.
        dst_crs: Target CRS. String.
        out_path: The path of the output file.
    """
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rio.open(out_path, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rio.band(src, i),
                destination=rio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.cubic)

def mask_A_by_B(A, B):
    """ Generate a mask from B, and applied it to A.
        All 0 values are excluded.
    """
    mask = B.sum(axis=0)>1e-3
    masked_A = mask*A
    return masked_A
```



## Reference

1. [Python GIS 超完整教程](https://automating-gis-processes.github.io/2016/course-info.html)
2. [Professor Zia's Personal Website](https://zia207.github.io/geospatial-python.io/)
3. [sentinel-2数据下载 大气校正 转ENVI格式](https://blog.csdn.net/sinat_28853941/article/details/78511167)
4. [03-SNAP处理Sentinel-2 L2A级数据（一）](https://blog.csdn.net/lidahuilidahui/article/details/102765420)
5. [UBC Course Notebook](https://clouds.eos.ubc.ca/~phil/courses/atsc301/html/rasterio_demo.html)
6. [利用Sen2cor对哨兵2号（Sentinel-2）L1C多光谱数据进行辐射定标和大气校正](https://zhuanlan.zhihu.com/p/31010043)
7. [Documentation of Sentinel Hub Python package](https://sentinelhub-py.readthedocs.io/en/latest/)
8. [sentinelhub-py](https://github.com/sentinel-hub/sentinelhub-py)