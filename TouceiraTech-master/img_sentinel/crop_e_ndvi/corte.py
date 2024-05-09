import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from shapely.geometry import mapping
import rioxarray as rxr
import xarray as xr
import geopandas as gpd

import earthpy as et
import earthpy.plot as ep

img = rxr.open_rasterio(r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\imgs\Janeiro2021\20012021\b8_2001.tif', masked=True)
img_salva = r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\imgs\Janeiro2021\20012021'
banda = 'b8'
data = '2001'
'''
ep.plot_bands(img,
               title="Imagem",
               cbar=False);
'''
shp = os.path.join(r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\shp\shp_area_total.shp')

shape = gpd.read_file(shp)

rct = img.rio.clip(shape.geometry.apply(mapping), shape.crs)
'''
ep.plot_bands(rct,    
              title="Area Total",
              cbar=False)
'''
caminho = os.path.join(img_salva, banda + '_' + data + '_recorte_areatotal.tif')

rct.rio.to_raster(caminho)

#########################################################

shp = os.path.join(r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\shp_P20_Infestado\shp_p20_infest.shp')

shape = gpd.read_file(shp)

rct = img.rio.clip(shape.geometry.apply(mapping), shape.crs)
'''
ep.plot_bands(rct,    
              title="P20 Infestado",
              cbar=False)
'''
caminho = os.path.join(img_salva, banda + '_' + data + '_recorte_p20infest.tif')

rct.rio.to_raster(caminho)

#########################################################

shp = os.path.join(r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\shp_P20_Mirapasto\shp_p20_mira.shp')

shape = gpd.read_file(shp)

rct = img.rio.clip(shape.geometry.apply(mapping), shape.crs)
'''
ep.plot_bands(rct,    
              title="P20 Mirapasto",
              cbar=False)
'''
caminho = os.path.join(img_salva, banda + '_' + data + '_recorte_p20mira.tif')

rct.rio.to_raster(caminho)

#########################################################

shp = os.path.join(r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\shp_P21_Infestado\shp_p21_infest.shp')

shape = gpd.read_file(shp)

rct = img.rio.clip(shape.geometry.apply(mapping), shape.crs)
'''
ep.plot_bands(rct,    
              title="P21 Infestado",
              cbar=False)
'''
caminho = os.path.join(img_salva, banda + '_' + data + '_recorte_p21infest.tif')

rct.rio.to_raster(caminho)

#########################################################

shp = os.path.join(r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\shp_P21_Mirapasto\shp_p21_mira.shp')

shape = gpd.read_file(shp)

rct = img.rio.clip(shape.geometry.apply(mapping), shape.crs)
'''
ep.plot_bands(rct,    
              title="P21 Mirapasto",
              cbar=False)
'''
caminho = os.path.join(img_salva, banda + '_' + data + '_recorte_p21mira.tif')

rct.rio.to_raster(caminho)

#########################################################

shp = os.path.join(r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\shp_g1_p20infest\shp_g1_p20infest.shp')

shape = gpd.read_file(shp)

rct = img.rio.clip(shape.geometry.apply(mapping), shape.crs)
'''
ep.plot_bands(rct,    
              title="G1 - P20Infestado",
              cbar=False)
'''
caminho = os.path.join(img_salva, banda + '_' + data + '_recorte_g1_p20infest.tif')

rct.rio.to_raster(caminho)

#########################################################

shp = os.path.join(r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\shp_g4_p20infest\shp_g4_p20infest.shp')

shape = gpd.read_file(shp)

rct = img.rio.clip(shape.geometry.apply(mapping), shape.crs)
'''
ep.plot_bands(rct,    
              title="G4 - P20Infestado",
              cbar=False)
'''
caminho = os.path.join(img_salva, banda + '_' + data + '_recorte_g4_p20infest.tif')

rct.rio.to_raster(caminho)

#########################################################

shp = os.path.join(r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\shp_g5_p20infest\shp_g5_p20infest.shp')

shape = gpd.read_file(shp)

rct = img.rio.clip(shape.geometry.apply(mapping), shape.crs)
'''
ep.plot_bands(rct,    
              title="G5 - P20Infestado",
              cbar=False)
'''
caminho = os.path.join(img_salva, banda + '_' + data + '_recorte_g5_p20infest.tif')

rct.rio.to_raster(caminho)

#########################################################

shp = os.path.join(r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\shp_g1_p20mira\shp_g1_p20mira.shp')

shape = gpd.read_file(shp)

rct = img.rio.clip(shape.geometry.apply(mapping), shape.crs)
'''
ep.plot_bands(rct,    
              title="G1 - P20Mirapasto",
              cbar=False)
'''
caminho = os.path.join(img_salva, banda + '_' + data + '_recorte_g1_p20mira.tif')

rct.rio.to_raster(caminho)

#########################################################

shp = os.path.join(r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\shp_g3_p20mira\shp_g3_p20mira.shp')


shape = gpd.read_file(shp)

rct = img.rio.clip(shape.geometry.apply(mapping), shape.crs)
'''
ep.plot_bands(rct,    
              title="G3 - P20Mirapasto",
              cbar=False)
'''
caminho = os.path.join(img_salva, banda + '_' + data + '_recorte_g3_p20mira.tif')

rct.rio.to_raster(caminho)

#########################################################

shp = os.path.join(r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\shp_g5_p20mira\shp_g5_p20mira.shp')

shape = gpd.read_file(shp)

rct = img.rio.clip(shape.geometry.apply(mapping), shape.crs)
'''
ep.plot_bands(rct,    
              title="G5 - P20Mirapasto",
              cbar=False)
'''
caminho = os.path.join(img_salva, banda + '_' + data + '_recorte_g5_p20mira.tif')

rct.rio.to_raster(caminho)

#########################################################

shp = os.path.join(r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\shp_g1_p21infest\shp_g1_p21infest.shp')

shape = gpd.read_file(shp)

rct = img.rio.clip(shape.geometry.apply(mapping), shape.crs)
'''
ep.plot_bands(rct,    
              title="G1 - P21Infestado",
              cbar=False)
'''
caminho = os.path.join(img_salva, banda + '_' + data + '_recorte_g1_p21infest.tif')

rct.rio.to_raster(caminho)

#########################################################

shp = os.path.join(r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\shp_g3_p21infest\shp_g3_p21infest.shp')

shape = gpd.read_file(shp)

rct = img.rio.clip(shape.geometry.apply(mapping), shape.crs)
'''
ep.plot_bands(rct,    
              title="G3 - P21Infestado",
              cbar=False)
'''
caminho = os.path.join(img_salva, banda + '_' + data + '_recorte_g3_p21infest.tif')

rct.rio.to_raster(caminho)

#########################################################

shp = os.path.join(r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\shp_g5_p21infest\shp_g5_p21infest.shp')

shape = gpd.read_file(shp)

rct = img.rio.clip(shape.geometry.apply(mapping), shape.crs)
'''
ep.plot_bands(rct,    
              title="G5 - P21Infestado",
              cbar=False)
'''
caminho = os.path.join(img_salva, banda + '_' + data + '_recorte_g5_p21infest.tif')

rct.rio.to_raster(caminho)

#########################################################

shp = os.path.join(r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\shp_g1_p21mira\shp_g1_p21mira.shp')

shape = gpd.read_file(shp)

rct = img.rio.clip(shape.geometry.apply(mapping), shape.crs)
'''
ep.plot_bands(rct,    
              title="G1 - P21Mirapasto",
              cbar=False)
'''
caminho = os.path.join(img_salva, banda + '_' + data + '_recorte_g1_p21mira.tif')

rct.rio.to_raster(caminho)

#########################################################

shp = os.path.join(r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\shp_g2_p21mira\shp_g2_p21mira.shp')

shape = gpd.read_file(shp)

rct = img.rio.clip(shape.geometry.apply(mapping), shape.crs)
'''
ep.plot_bands(rct,    
              title="G2 - P21Mirapasto",
              cbar=False)
'''
caminho = os.path.join(img_salva, banda + '_' + data + '_recorte_g2_p21mira.tif')

rct.rio.to_raster(caminho)

#########################################################

shp = os.path.join(r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\shp_g4_p21mira\shp_g4_p21mira.shp')

shape = gpd.read_file(shp)

rct = img.rio.clip(shape.geometry.apply(mapping), shape.crs)
'''
ep.plot_bands(rct,    
              title="G4 - P21Mirapasto",
              cbar=False)
'''
caminho = os.path.join(img_salva, banda + '_' + data + '_recorte_g4_p21mira.tif')

rct.rio.to_raster(caminho)