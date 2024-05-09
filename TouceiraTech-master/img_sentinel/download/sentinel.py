from sentinelsat.sentinel import SentinelAPI
from sentinelsat import read_geojson, geojson_to_wkt

s2_api = SentinelAPI(
    user=user,
    password=password,
    api_url="https://scihub.copernicus.eu/apihub/"
)

products = s2_api.query(
    area = geojson_to_wkt(read_geojson('area.geojson')),
    date = ("20210101", "20210110"),
    platformname = "Sentinel-2",
    platformserialidentifier = "Sentinel-2A",
    producttype = "S2MSI2A")
    #cloudcoverpercentage="(0,40)"

#Para ver todos os produtos gerados por essa query
print(products)
'''
#download
gj = s2_api.to_geojson(products)
s2_api.download_all(products)
'''