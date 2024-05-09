import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
import numpy as np
import rasterio as rio
import geopandas as gpd
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import psycopg2
import pickle
from matplotlib import cm

#Comando para converter jp2 para tif
#for %i in (*.jp2) do gdal_translate -of GTiff %i %~ni.tif

caminho = r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\imgs\Janeiro2021\20012021'
caminho_imgrgb = os.path.join(caminho, 'Resultados', 'NDVI')
caminho_hist = os.path.join(caminho, 'Resultados', 'Histograma')

data_img = '2001' 
data = '2021-01-20'
id = 562 #Verificar o último id que consta no tabela no banco de dados, sempre que realizar uma inserção no banco somar 16 a esse valor

cmh_b4_areatotal = os.path.join(caminho, 'b4_' + data_img + '_recorte_areatotal.tif')
cmh_b8_areatotal = os.path.join(caminho, 'b8_' + data_img + '_recorte_areatotal.tif')
cmh_b4_p20infest = os.path.join(caminho, 'b4_' + data_img + '_recorte_p20infest.tif')
cmh_b8_p20infest = os.path.join(caminho, 'b8_' + data_img + '_recorte_p20infest.tif')
cmh_b4_p20mira = os.path.join(caminho, 'b4_' + data_img + '_recorte_p20mira.tif')
cmh_b8_p20mira = os.path.join(caminho, 'b8_' + data_img + '_recorte_p20mira.tif')
cmh_b4_p21infest = os.path.join(caminho, 'b4_' + data_img + '_recorte_p21infest.tif')
cmh_b8_p21infest = os.path.join(caminho, 'b8_' + data_img + '_recorte_p21infest.tif')
cmh_b4_p21mira = os.path.join(caminho, 'b4_' + data_img + '_recorte_p21mira.tif')
cmh_b8_p21mira = os.path.join(caminho, 'b8_' + data_img + '_recorte_p21mira.tif')
cmh_b4_g1_p20infest = os.path.join(caminho, 'b4_' + data_img + '_recorte_g1_p20infest.tif')
cmh_b8_g1_p20infest = os.path.join(caminho, 'b8_' + data_img + '_recorte_g1_p20infest.tif')
cmh_b4_g4_p20infest = os.path.join(caminho, 'b4_' + data_img + '_recorte_g4_p20infest.tif')
cmh_b8_g4_p20infest = os.path.join(caminho, 'b8_' + data_img + '_recorte_g4_p20infest.tif')
cmh_b4_g5_p20infest = os.path.join(caminho, 'b4_' + data_img + '_recorte_g5_p20infest.tif')
cmh_b8_g5_p20infest = os.path.join(caminho, 'b8_' + data_img + '_recorte_g5_p20infest.tif')
cmh_b4_g1_p20mira = os.path.join(caminho, 'b4_' + data_img + '_recorte_g1_p20mira.tif')
cmh_b8_g1_p20mira = os.path.join(caminho, 'b8_' + data_img + '_recorte_g1_p20mira.tif')
cmh_b4_g3_p20mira = os.path.join(caminho, 'b4_' + data_img + '_recorte_g3_p20mira.tif')
cmh_b8_g3_p20mira = os.path.join(caminho, 'b8_' + data_img + '_recorte_g3_p20mira.tif')
cmh_b4_g5_p20mira = os.path.join(caminho, 'b4_' + data_img + '_recorte_g5_p20mira.tif')
cmh_b8_g5_p20mira = os.path.join(caminho, 'b8_' + data_img + '_recorte_g5_p20mira.tif')
cmh_b4_g1_p21infest = os.path.join(caminho, 'b4_' + data_img + '_recorte_g1_p21infest.tif')
cmh_b8_g1_p21infest = os.path.join(caminho, 'b8_' + data_img + '_recorte_g1_p21infest.tif')
cmh_b4_g3_p21infest = os.path.join(caminho, 'b4_' + data_img + '_recorte_g3_p21infest.tif')
cmh_b8_g3_p21infest = os.path.join(caminho, 'b8_' + data_img + '_recorte_g3_p21infest.tif')
cmh_b4_g5_p21infest = os.path.join(caminho, 'b4_' + data_img + '_recorte_g5_p21infest.tif')
cmh_b8_g5_p21infest = os.path.join(caminho, 'b8_' + data_img + '_recorte_g5_p21infest.tif')
cmh_b4_g1_p21mira = os.path.join(caminho, 'b4_' + data_img + '_recorte_g1_p21mira.tif')
cmh_b8_g1_p21mira = os.path.join(caminho, 'b8_' + data_img + '_recorte_g1_p21mira.tif')
cmh_b4_g2_p21mira = os.path.join(caminho, 'b4_' + data_img + '_recorte_g2_p21mira.tif')
cmh_b8_g2_p21mira = os.path.join(caminho, 'b8_' + data_img + '_recorte_g2_p21mira.tif')
cmh_b4_g4_p21mira = os.path.join(caminho, 'b4_' + data_img + '_recorte_g4_p21mira.tif')
cmh_b8_g4_p21mira = os.path.join(caminho, 'b8_' + data_img + '_recorte_g4_p21mira.tif')


#Conecta com o BD
conn = psycopg2.connect("host='localhost' port='5432' dbname='db_2019' user='postgres' password='12345'")
cur = conn.cursor()

################################################ÁREA TOTAL##############################################################

with rio.open(cmh_b4_areatotal) as src:
    naip_b4 = src.read()

with rio.open(cmh_b8_areatotal) as src:
    naip_b8 = src.read()

naip_ndvi = es.normalized_diff(naip_b8, naip_b4)

with rio.open(cmh_b8_areatotal) as src:
    naip_data = src.read()
    naip_meta = src.profile

naip_transform = naip_meta["transform"]
naip_crs = naip_meta["crs"]

naip_meta['dtype'] = "float64"

imagem = naip_ndvi.squeeze()
xmin, xmax, ymin, ymax = (0, 50, 0, 75)
imshow_kwargs = {
    'vmax': 1,
    'vmin': -1,
    'cmap': 'RdYlGn',
    'extent': (xmin, xmax, ymin, ymax),
}

plt.imshow(imagem, **imshow_kwargs)
plt.colorbar()
plt.title("NDVI - Area Total" + " " + "-" + " " + data)
plt.savefig(caminho_imgrgb + '\AreaTotal.png')
plt.close()

ep.hist(naip_ndvi,
        figsize=(12, 6),
        title=["Histograma - Area Total" + " " + "-" + " " + data])
plt.savefig(caminho_hist + '\AreaTotal.png')
plt.close()

media = naip_ndvi.mean()
desvio = naip_ndvi.std()
variancia = naip_ndvi.var()
idpotreiro = None
idsubarea = None
area = 'NDVI Area Total'

query =  "INSERT INTO ndvi (id, idpotreiro, data, ndvi_medio, ndvi_desviop, ndvi_variance, descricao, idsubarea) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
values = (id, idpotreiro, data, media, desvio, variancia, area, idsubarea)

cur.execute(query, values)
conn.commit()

print("Inseriu")

naip_ndvi_outpath = os.path.join(caminho, 'recorte_ndvi_' + data_img + '_areatotal.tif')
 
# Write your the ndvi raster object
with rio.open(naip_ndvi_outpath, 'w', **naip_meta) as dst:
    dst.write(naip_ndvi)

################################################ÁREA TOTAL##############################################################

################################################P20 INFESTADO###########################################################

with rio.open(cmh_b4_p20infest) as src:
    naip_b4 = src.read()

with rio.open(cmh_b8_p20infest) as src:
    naip_b8 = src.read()

naip_ndvi = es.normalized_diff(naip_b8, naip_b4)

with rio.open(cmh_b8_p20infest) as src:
    naip_data = src.read()
    naip_meta = src.profile

naip_transform = naip_meta["transform"]
naip_crs = naip_meta["crs"]
naip_meta['dtype'] = "float64"

imagem = naip_ndvi.squeeze()
xmin, xmax, ymin, ymax = (0, 50, 0, 75)
imshow_kwargs = {
    'vmax': 1,
    'vmin': -1,
    'cmap': 'RdYlGn',
    'extent': (xmin, xmax, ymin, ymax),
}

plt.imshow(imagem, **imshow_kwargs)
plt.colorbar()
plt.title("NDVI - P20 Infestado" + " " + "-" + " " + data)
plt.savefig(caminho_imgrgb + '\p20infest.png')
plt.close()

ep.hist(naip_ndvi,
        figsize=(12, 6),
        title=["Histograma - P20 Infestado" + " " + "-" + " " + data])
plt.savefig(caminho_hist + '\p20infest.png')
plt.close()

media = naip_ndvi.mean()
desvio = naip_ndvi.std()
variancia = naip_ndvi.var()
idpotreiro = 1
idsubarea = None
area = 'NDVI P20 Infestado'
id1 = id + 1

query =  "INSERT INTO ndvi (id, idpotreiro, data, ndvi_medio, ndvi_desviop, ndvi_variance, descricao, idsubarea) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
values = (id1, idpotreiro, data, media, desvio, variancia, area, idsubarea)

cur.execute(query, values)
conn.commit()

print("Inseriu")

naip_ndvi_outpath = os.path.join(caminho, 'recorte_ndvi_' + data_img + '_p20infest.tif')
 
# Write your the ndvi raster object
with rio.open(naip_ndvi_outpath, 'w', **naip_meta) as dst:
    dst.write(naip_ndvi)

################################################P20 INFESTADO###########################################################

################################################P20 MIRAPASTO###########################################################

with rio.open(cmh_b4_p20mira) as src:
    naip_b4 = src.read()

with rio.open(cmh_b8_p20mira) as src:
    naip_b8 = src.read()

naip_ndvi = es.normalized_diff(naip_b8, naip_b4)

with rio.open(cmh_b8_p20mira) as src:
    naip_data = src.read()
    naip_meta = src.profile

naip_transform = naip_meta["transform"]
naip_crs = naip_meta["crs"]
naip_meta['dtype'] = "float64"

imagem = naip_ndvi.squeeze()
xmin, xmax, ymin, ymax = (0, 35, 0, 60)
imshow_kwargs = {
    'vmax': 1,
    'vmin': -1,
    'cmap': 'RdYlGn',
    'extent': (xmin, xmax, ymin, ymax),
}

plt.imshow(imagem, **imshow_kwargs)
plt.colorbar()
plt.title("NDVI - P20 Mirapasto" + " " + "-" + " " + data)
plt.savefig(caminho_imgrgb + '\p20mira.png')
plt.close()

ep.hist(naip_ndvi,
        figsize=(12, 6),
        title=["Histograma - P20 Mirapasto" + " " + "-" + " " + data])
plt.savefig(caminho_hist + '\p20mira.png')
plt.close()

media = naip_ndvi.mean()
desvio = naip_ndvi.std()
variancia = naip_ndvi.var()
idpotreiro = 2
idsubarea = None
area = 'NDVI P20 Mirapasto'
id2 = id + 2

query =  "INSERT INTO ndvi (id, idpotreiro, data, ndvi_medio, ndvi_desviop, ndvi_variance, descricao, idsubarea) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
values = (id2, idpotreiro, data, media, desvio, variancia, area, idsubarea)

cur.execute(query, values)
conn.commit()

print("Inseriu")

naip_ndvi_outpath = os.path.join(caminho, 'recorte_ndvi_' + data_img + '_p20mira.tif')
 
# Write your the ndvi raster object
with rio.open(naip_ndvi_outpath, 'w', **naip_meta) as dst:
    dst.write(naip_ndvi)

################################################P20 MIRAPASTO###########################################################

################################################P21 INFESTADO###########################################################

with rio.open(cmh_b4_p21infest) as src:
    naip_b4 = src.read()

with rio.open(cmh_b8_p21infest) as src:
    naip_b8 = src.read()

naip_ndvi = es.normalized_diff(naip_b8, naip_b4)

with rio.open(cmh_b8_p21infest) as src:
    naip_data = src.read()
    naip_meta = src.profile

naip_transform = naip_meta["transform"]
naip_crs = naip_meta["crs"]
naip_meta['dtype'] = "float64"

imagem = naip_ndvi.squeeze()
xmin, xmax, ymin, ymax = (0, 35, 0, 60)
imshow_kwargs = {
    'vmax': 1,
    'vmin': -1,
    'cmap': 'RdYlGn',
    'extent': (xmin, xmax, ymin, ymax),
}

plt.imshow(imagem, **imshow_kwargs)
plt.colorbar()
plt.title("NDVI - P21 Infestado" + " " + "-" + " " + data)
plt.savefig(caminho_imgrgb + '\p21infest.png')
plt.close()

ep.hist(naip_ndvi,
        figsize=(12, 6),
        title=["Histograma - P21 Infestado" + " " + "-" + " " + data])
plt.savefig(caminho_hist + '\p21infest.png')
plt.close()

media = naip_ndvi.mean()
desvio = naip_ndvi.std()
variancia = naip_ndvi.var()
idpotreiro = 3
idsubarea = None
area = 'NDVI P21 Infestado'
id3 = id + 3

query =  "INSERT INTO ndvi (id, idpotreiro, data, ndvi_medio, ndvi_desviop, ndvi_variance, descricao, idsubarea) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
values = (id3, idpotreiro, data, media, desvio, variancia, area, idsubarea)

cur.execute(query, values)
conn.commit()

print("Inseriu")

naip_ndvi_outpath = os.path.join(caminho, 'recorte_ndvi_' + data_img + '_p21infest.tif')
 
# Write your the ndvi raster object
with rio.open(naip_ndvi_outpath, 'w', **naip_meta) as dst:
    dst.write(naip_ndvi)

################################################P21 INFESTADO###########################################################

################################################P21 MIRAPASTO###########################################################

with rio.open(cmh_b4_p21mira) as src:
    naip_b4 = src.read()

with rio.open(cmh_b8_p21mira) as src:
    naip_b8 = src.read()

naip_ndvi = es.normalized_diff(naip_b8, naip_b4)

with rio.open(cmh_b8_p21mira) as src:
    naip_data = src.read()
    naip_meta = src.profile

naip_transform = naip_meta["transform"]
naip_crs = naip_meta["crs"]
naip_meta['dtype'] = "float64"

imagem = naip_ndvi.squeeze()
xmin, xmax, ymin, ymax = (0, 35, 0, 60)
imshow_kwargs = {
    'vmax': 1,
    'vmin': -1,
    'cmap': 'RdYlGn',
    'extent': (xmin, xmax, ymin, ymax),
}

plt.imshow(imagem, **imshow_kwargs)
plt.colorbar()
plt.title("NDVI - P21 Mirapasto" + " " + "-" + " " + data)
plt.savefig(caminho_imgrgb + '\p21mira.png')
plt.close()

ep.hist(naip_ndvi,
        figsize=(12, 6),
        title=["Histograma - P21 Mirapasto" + " " + "-" + " " + data])
plt.savefig(caminho_hist + '\p21mira.png')
plt.close()

media = naip_ndvi.mean()
desvio = naip_ndvi.std()
variancia = naip_ndvi.var()
idpotreiro = 4
idsubarea = None
area = 'NDVI P21 Mirapasto'
id4 = id + 4

query =  "INSERT INTO ndvi (id, idpotreiro, data, ndvi_medio, ndvi_desviop, ndvi_variance, descricao, idsubarea) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
values = (id4, idpotreiro, data, media, desvio, variancia, area, idsubarea)

cur.execute(query, values)
conn.commit()

print("Inseriu")

naip_ndvi_outpath = os.path.join(caminho, 'recorte_ndvi_' + data_img + '_p21mira.tif')
 
# Write your the ndvi raster object
with rio.open(naip_ndvi_outpath, 'w', **naip_meta) as dst:
    dst.write(naip_ndvi)

################################################P21 MIRAPASTO###########################################################

################################################GAIOLA 1 P20 INFESTADO##################################################

with rio.open(cmh_b4_g1_p20infest) as src:
    naip_b4 = src.read()

with rio.open(cmh_b8_g1_p20infest) as src:
    naip_b8 = src.read()

naip_ndvi = es.normalized_diff(naip_b8, naip_b4)

with rio.open(cmh_b8_g1_p20infest) as src:
    naip_data = src.read()
    naip_meta = src.profile

naip_transform = naip_meta["transform"]
naip_crs = naip_meta["crs"]
naip_meta['dtype'] = "float64"

imagem = naip_ndvi.squeeze()
xmin, xmax, ymin, ymax = (0, 35, 0, 60)
imshow_kwargs = {
    'vmax': 1,
    'vmin': -1,
    'cmap': 'RdYlGn',
    'extent': (xmin, xmax, ymin, ymax),
}

plt.imshow(imagem, **imshow_kwargs)
plt.colorbar()
plt.title("NDVI - Gaiola 1 P20 Infestado" + " " + "-" + " " + data)
plt.savefig(caminho_imgrgb + '\g1_p20infest.png')
plt.close()

ep.hist(naip_ndvi,
        figsize=(12, 6),
        title=["Histograma - Gaiola 1 P20 Infestado" + " " + "-" + " " + data])
plt.savefig(caminho_hist + '\g1_p20infest.png')
plt.close()

media = naip_ndvi.mean()
desvio = naip_ndvi.std()
variancia = naip_ndvi.var()
idpotreiro = 1
idsubarea = 25
area = 'NDVI Gaiola 1 P20 Infestado'
id5 = id + 5

query =  "INSERT INTO ndvi (id, idpotreiro, data, ndvi_medio, ndvi_desviop, ndvi_variance, descricao, idsubarea) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
values = (id5, idpotreiro, data, media, desvio, variancia, area, idsubarea)

cur.execute(query, values)
conn.commit()

print("Inseriu")

naip_ndvi_outpath = os.path.join(caminho, 'recorte_ndvi_' + data_img + '_g1_p20infest.tif')
 
# Write your the ndvi raster object
with rio.open(naip_ndvi_outpath, 'w', **naip_meta) as dst:
    dst.write(naip_ndvi)

################################################GAIOLA 1 P20 INFESTADO##################################################

################################################GAIOLA 4 P20 INFESTADO##################################################

with rio.open(cmh_b4_g4_p20infest) as src:
    naip_b4 = src.read()

with rio.open(cmh_b8_g4_p20infest) as src:
    naip_b8 = src.read()

naip_ndvi = es.normalized_diff(naip_b8, naip_b4)

with rio.open(cmh_b8_g4_p20infest) as src:
    naip_data = src.read()
    naip_meta = src.profile

naip_transform = naip_meta["transform"]
naip_crs = naip_meta["crs"]
naip_meta['dtype'] = "float64"

imagem = naip_ndvi.squeeze()
xmin, xmax, ymin, ymax = (0, 35, 0, 60)
imshow_kwargs = {
    'vmax': 1,
    'vmin': -1,
    'cmap': 'RdYlGn',
    'extent': (xmin, xmax, ymin, ymax),
}

plt.imshow(imagem, **imshow_kwargs)
plt.colorbar()
plt.title("NDVI - Gaiola 4 P20 Infestado" + " " + "-" + " " + data)
plt.savefig(caminho_imgrgb + '\g4_p20infest.png')
plt.close()

ep.hist(naip_ndvi,
        figsize=(12, 6),
        title=["Histograma - Gaiola 4 P20 Infestado" + " " + "-" + " " + data])
plt.savefig(caminho_hist + '\g4_p20infest.png')
plt.close()

media = naip_ndvi.mean()
desvio = naip_ndvi.std()
variancia = naip_ndvi.var()
idpotreiro = 1
idsubarea = 26
area = 'NDVI Gaiola 4 P20 Infestado'
id6 = id + 6

query =  "INSERT INTO ndvi (id, idpotreiro, data, ndvi_medio, ndvi_desviop, ndvi_variance, descricao, idsubarea) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
values = (id6, idpotreiro, data, media, desvio, variancia, area, idsubarea)

cur.execute(query, values)
conn.commit()

print("Inseriu")

naip_ndvi_outpath = os.path.join(caminho, 'recorte_ndvi_' + data_img + '_g4_p20infest.tif')
 
# Write your the ndvi raster object
with rio.open(naip_ndvi_outpath, 'w', **naip_meta) as dst:
    dst.write(naip_ndvi)

################################################GAIOLA 4 P20 INFESTADO##################################################

################################################GAIOLA 5 P20 INFESTADO##################################################

with rio.open(cmh_b4_g5_p20infest) as src:
    naip_b4 = src.read()

with rio.open(cmh_b8_g5_p20infest) as src:
    naip_b8 = src.read()

naip_ndvi = es.normalized_diff(naip_b8, naip_b4)

with rio.open(cmh_b8_g5_p20infest) as src:
    naip_data = src.read()
    naip_meta = src.profile

naip_transform = naip_meta["transform"]
naip_crs = naip_meta["crs"]
naip_meta['dtype'] = "float64"

imagem = naip_ndvi.squeeze()
xmin, xmax, ymin, ymax = (0, 35, 0, 60)
imshow_kwargs = {
    'vmax': 1,
    'vmin': -1,
    'cmap': 'RdYlGn',
    'extent': (xmin, xmax, ymin, ymax),
}

plt.imshow(imagem, **imshow_kwargs)
plt.colorbar()
plt.title("NDVI - Gaiola 5 P20 Infestado" + " " + "-" + " " + data)
plt.savefig(caminho_imgrgb + '\g5_p20infest.png')
plt.close()

ep.hist(naip_ndvi,
        figsize=(12, 6),
        title=["Histograma - Gaiola 5 P20 Infestado" + " " + "-" + " " + data])
plt.savefig(caminho_hist + '\g5_p20infest.png')
plt.close()

media = naip_ndvi.mean()
desvio = naip_ndvi.std()
variancia = naip_ndvi.var()
idpotreiro = 1
idsubarea = 27
area = 'NDVI Gaiola 5 P20 Infestado'
id7 = id + 7

query =  "INSERT INTO ndvi (id, idpotreiro, data, ndvi_medio, ndvi_desviop, ndvi_variance, descricao, idsubarea) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
values = (id7, idpotreiro, data, media, desvio, variancia, area, idsubarea)

cur.execute(query, values)
conn.commit()

print("Inseriu")

naip_ndvi_outpath = os.path.join(caminho, 'recorte_ndvi_' + data_img + '_g5_p20infest.tif')
 
# Write your the ndvi raster object
with rio.open(naip_ndvi_outpath, 'w', **naip_meta) as dst:
    dst.write(naip_ndvi)

################################################GAIOLA 5 P20 INFESTADO##################################################

################################################GAIOLA 1 P20 MIRAPASTO##################################################

with rio.open(cmh_b4_g1_p20mira) as src:
    naip_b4 = src.read()

with rio.open(cmh_b8_g1_p20mira) as src:
    naip_b8 = src.read()

naip_ndvi = es.normalized_diff(naip_b8, naip_b4)

with rio.open(cmh_b8_g1_p20mira) as src:
    naip_data = src.read()
    naip_meta = src.profile

naip_transform = naip_meta["transform"]
naip_crs = naip_meta["crs"]
naip_meta['dtype'] = "float64"

imagem = naip_ndvi.squeeze()
xmin, xmax, ymin, ymax = (0, 35, 0, 60)
imshow_kwargs = {
    'vmax': 1,
    'vmin': -1,
    'cmap': 'RdYlGn',
    'extent': (xmin, xmax, ymin, ymax),
}

plt.imshow(imagem, **imshow_kwargs)
plt.colorbar()
plt.title("NDVI - Gaiola 1 P20 Mirapasto" + " " + "-" + " " + data)
plt.savefig(caminho_imgrgb + '\g1_p20mira.png')
plt.close()

ep.hist(naip_ndvi,
        figsize=(12, 6),
        title=["Histograma - Gaiola 1 P20 Mirapasto" + " " + "-" + " " + data])
plt.savefig(caminho_hist + '\g1_p20mira.png')
plt.close()

media = naip_ndvi.mean()
desvio = naip_ndvi.std()
variancia = naip_ndvi.var()
idpotreiro = 2
idsubarea = 28
area = 'NDVI Gaiola 1 P20 Mirapasto'
id8 = id + 8

query =  "INSERT INTO ndvi (id, idpotreiro, data, ndvi_medio, ndvi_desviop, ndvi_variance, descricao, idsubarea) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
values = (id8, idpotreiro, data, media, desvio, variancia, area, idsubarea)

cur.execute(query, values)
conn.commit()

print("Inseriu")

naip_ndvi_outpath = os.path.join(caminho, 'recorte_ndvi_' + data_img + '_g1_p20mira.tif')
 
# Write your the ndvi raster object
with rio.open(naip_ndvi_outpath, 'w', **naip_meta) as dst:
    dst.write(naip_ndvi)

################################################GAIOLA 1 P20 MIRASPASTO##################################################

################################################GAIOLA 3 P20 MIRAPASTO##################################################

with rio.open(cmh_b4_g3_p20mira) as src:
    naip_b4 = src.read()

with rio.open(cmh_b8_g3_p20mira) as src:
    naip_b8 = src.read()

naip_ndvi = es.normalized_diff(naip_b8, naip_b4)

with rio.open(cmh_b8_g3_p20mira) as src:
    naip_data = src.read()
    naip_meta = src.profile

naip_transform = naip_meta["transform"]
naip_crs = naip_meta["crs"]
naip_meta['dtype'] = "float64"

imagem = naip_ndvi.squeeze()
xmin, xmax, ymin, ymax = (0, 35, 0, 60)
imshow_kwargs = {
    'vmax': 1,
    'vmin': -1,
    'cmap': 'RdYlGn',
    'extent': (xmin, xmax, ymin, ymax),
}

plt.imshow(imagem, **imshow_kwargs)
plt.colorbar()
plt.title("NDVI - Gaiola 3 P20 Mirapasto" + " " + "-" + " " + data)
plt.savefig(caminho_imgrgb + '\g3_p20mira.png')
plt.close()

ep.hist(naip_ndvi,
        figsize=(12, 6),
        title=["Histograma - Gaiola 3 P20 Mirapasto" + " " + "-" + " " + data])
plt.savefig(caminho_hist + '\g3_p20mira.png')
plt.close()

media = naip_ndvi.mean()
desvio = naip_ndvi.std()
variancia = naip_ndvi.var()
idpotreiro = 2
idsubarea = 29
area = 'NDVI Gaiola 3 P20 Mirapasto'
id9 = id + 9

query =  "INSERT INTO ndvi (id, idpotreiro, data, ndvi_medio, ndvi_desviop, ndvi_variance, descricao, idsubarea) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
values = (id9, idpotreiro, data, media, desvio, variancia, area, idsubarea)

cur.execute(query, values)
conn.commit()

print("Inseriu")

naip_ndvi_outpath = os.path.join(caminho, 'recorte_ndvi_' + data_img + '_g3_p20mira.tif')
 
# Write your the ndvi raster object
with rio.open(naip_ndvi_outpath, 'w', **naip_meta) as dst:
    dst.write(naip_ndvi)

################################################GAIOLA 3 P20 MIRAPASTO##################################################

################################################GAIOLA 5 P20 MIRAPASTO##################################################

with rio.open(cmh_b4_g5_p20mira) as src:
    naip_b4 = src.read()

with rio.open(cmh_b8_g5_p20mira) as src:
    naip_b8 = src.read()

naip_ndvi = es.normalized_diff(naip_b8, naip_b4)

with rio.open(cmh_b8_g5_p20mira) as src:
    naip_data = src.read()
    naip_meta = src.profile

naip_transform = naip_meta["transform"]
naip_crs = naip_meta["crs"]
naip_meta['dtype'] = "float64"

imagem = naip_ndvi.squeeze()
xmin, xmax, ymin, ymax = (0, 35, 0, 60)
imshow_kwargs = {
    'vmax': 1,
    'vmin': -1,
    'cmap': 'RdYlGn',
    'extent': (xmin, xmax, ymin, ymax),
}

plt.imshow(imagem, **imshow_kwargs)
plt.colorbar()
plt.title("NDVI - Gaiola 5 P20 Mirapasto" + " " + "-" + " " + data)
plt.savefig(caminho_imgrgb + '\g5_p20mira.png')
plt.close()

ep.hist(naip_ndvi,
        figsize=(12, 6),
        title=["Histograma - Gaiola 5 P20 Mirapasto" + " " + "-" + " " + data])
plt.savefig(caminho_hist + '\g5_p20mira.png')
plt.close()

media = naip_ndvi.mean()
desvio = naip_ndvi.std()
variancia = naip_ndvi.var()
idpotreiro = 2
idsubarea = 30
area = 'NDVI Gaiola 5 P20 Mirapasto'
id10 = id + 10

query =  "INSERT INTO ndvi (id, idpotreiro, data, ndvi_medio, ndvi_desviop, ndvi_variance, descricao, idsubarea) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
values = (id10, idpotreiro, data, media, desvio, variancia, area, idsubarea)

cur.execute(query, values)
conn.commit()

print("Inseriu")

naip_ndvi_outpath = os.path.join(caminho, 'recorte_ndvi_' + data_img + '_g5_p20mira.tif')
 
# Write your the ndvi raster object
with rio.open(naip_ndvi_outpath, 'w', **naip_meta) as dst:
    dst.write(naip_ndvi)

################################################GAIOLA 5 P20 MIRAPASTO##################################################

################################################GAIOLA 1 P21 INFESTADO##################################################

with rio.open(cmh_b4_g1_p21infest) as src:
    naip_b4 = src.read()

with rio.open(cmh_b8_g1_p21infest) as src:
    naip_b8 = src.read()

naip_ndvi = es.normalized_diff(naip_b8, naip_b4)

with rio.open(cmh_b8_g1_p21infest) as src:
    naip_data = src.read()
    naip_meta = src.profile

naip_transform = naip_meta["transform"]
naip_crs = naip_meta["crs"]
naip_meta['dtype'] = "float64"

imagem = naip_ndvi.squeeze()
xmin, xmax, ymin, ymax = (0, 35, 0, 60)
imshow_kwargs = {
    'vmax': 1,
    'vmin': -1,
    'cmap': 'RdYlGn',
    'extent': (xmin, xmax, ymin, ymax),
}

plt.imshow(imagem, **imshow_kwargs)
plt.colorbar()
plt.title("NDVI - Gaiola 1 P21 Infestado" + " " + "-" + " " + data)
plt.savefig(caminho_imgrgb + '\g1_p21infest.png')
plt.close()

ep.hist(naip_ndvi,
        figsize=(12, 6),
        title=["Histograma - Gaiola 1 P21 Infestado" + " " + "-" + " " + data])
plt.savefig(caminho_hist + '\g1_p21infest.png')
plt.close()

media = naip_ndvi.mean()
desvio = naip_ndvi.std()
variancia = naip_ndvi.var()
idpotreiro = 3
idsubarea = 31
area = 'NDVI Gaiola 1 P21 Infestado'
id11 = id + 11

query =  "INSERT INTO ndvi (id, idpotreiro, data, ndvi_medio, ndvi_desviop, ndvi_variance, descricao, idsubarea) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
values = (id11, idpotreiro, data, media, desvio, variancia, area, idsubarea)

cur.execute(query, values)
conn.commit()

print("Inseriu")

naip_ndvi_outpath = os.path.join(caminho, 'recorte_ndvi_' + data_img + '_g1_p21infest.tif')
 
# Write your the ndvi raster object
with rio.open(naip_ndvi_outpath, 'w', **naip_meta) as dst:
    dst.write(naip_ndvi)

################################################GAIOLA 1 P21 INFESTADO##################################################

################################################GAIOLA 3 P21 INFESTADO##################################################

with rio.open(cmh_b4_g3_p21infest) as src:
    naip_b4 = src.read()

with rio.open(cmh_b8_g3_p21infest) as src:
    naip_b8 = src.read()

naip_ndvi = es.normalized_diff(naip_b8, naip_b4)

with rio.open(cmh_b8_g3_p21infest) as src:
    naip_data = src.read()
    naip_meta = src.profile

naip_transform = naip_meta["transform"]
naip_crs = naip_meta["crs"]
naip_meta['dtype'] = "float64"

imagem = naip_ndvi.squeeze()
xmin, xmax, ymin, ymax = (0, 35, 0, 60)
imshow_kwargs = {
    'vmax': 1,
    'vmin': -1,
    'cmap': 'RdYlGn',
    'extent': (xmin, xmax, ymin, ymax),
}

plt.imshow(imagem, **imshow_kwargs)
plt.colorbar()
plt.title("NDVI - Gaiola 3 P21 Infestado" + " " + "-" + " " + data)
plt.savefig(caminho_imgrgb + '\g3_p21infest.png')
plt.close()

ep.hist(naip_ndvi,
        figsize=(12, 6),
        title=["Histograma - Gaiola 3 P21 Infestado" + " " + "-" + " " + data])
plt.savefig(caminho_hist + '\g3_p21infest.png')
plt.close()

media = naip_ndvi.mean()
desvio = naip_ndvi.std()
variancia = naip_ndvi.var()
idpotreiro = 3
idsubarea = 32
area = 'NDVI Gaiola 3 P21 Infestado'
id12 = id + 12

query =  "INSERT INTO ndvi (id, idpotreiro, data, ndvi_medio, ndvi_desviop, ndvi_variance, descricao, idsubarea) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
values = (id12, idpotreiro, data, media, desvio, variancia, area, idsubarea)

cur.execute(query, values)
conn.commit()

print("Inseriu")

naip_ndvi_outpath = os.path.join(caminho, 'recorte_ndvi_' + data_img + '_g3_p21infest.tif')
 
# Write your the ndvi raster object
with rio.open(naip_ndvi_outpath, 'w', **naip_meta) as dst:
    dst.write(naip_ndvi)

################################################GAIOLA 3 P21 INFESTADO##################################################

################################################GAIOLA 5 P21 INFESTADO##################################################

with rio.open(cmh_b4_g5_p21infest) as src:
    naip_b4 = src.read()

with rio.open(cmh_b8_g5_p21infest) as src:
    naip_b8 = src.read()

naip_ndvi = es.normalized_diff(naip_b8, naip_b4)

with rio.open(cmh_b8_g5_p21infest) as src:
    naip_data = src.read()
    naip_meta = src.profile

naip_transform = naip_meta["transform"]
naip_crs = naip_meta["crs"]
naip_meta['dtype'] = "float64"

imagem = naip_ndvi.squeeze()
xmin, xmax, ymin, ymax = (0, 35, 0, 60)
imshow_kwargs = {
    'vmax': 1,
    'vmin': -1,
    'cmap': 'RdYlGn',
    'extent': (xmin, xmax, ymin, ymax),
}

plt.imshow(imagem, **imshow_kwargs)
plt.colorbar()
plt.title("NDVI - Gaiola 5 P21 Infestado" + " " + "-" + " " + data)
plt.savefig(caminho_imgrgb + '\g5_p21infest.png')
plt.close()

ep.hist(naip_ndvi,
        figsize=(12, 6),
        title=["Histograma - Gaiola 5 P21 Infestado" + " " + "-" + " " + data])
plt.savefig(caminho_hist + '\g5_p21infest.png')
plt.close()

media = naip_ndvi.mean()
desvio = naip_ndvi.std()
variancia = naip_ndvi.var()
idpotreiro = 3
idsubarea = 33
area = 'NDVI Gaiola 5 P21 Infestado'
id13 = id + 13

query =  "INSERT INTO ndvi (id, idpotreiro, data, ndvi_medio, ndvi_desviop, ndvi_variance, descricao, idsubarea) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
values = (id13, idpotreiro, data, media, desvio, variancia, area, idsubarea)

cur.execute(query, values)
conn.commit()

print("Inseriu")

naip_ndvi_outpath = os.path.join(caminho, 'recorte_ndvi_' + data_img + '_g5_p21infest.tif')
 
# Write your the ndvi raster object
with rio.open(naip_ndvi_outpath, 'w', **naip_meta) as dst:
    dst.write(naip_ndvi)

################################################GAIOLA 5 P21 INFESTADO##################################################

################################################GAIOLA 1 P21 MIRAPASTO##################################################

with rio.open(cmh_b4_g1_p21mira) as src:
    naip_b4 = src.read()

with rio.open(cmh_b8_g1_p21mira) as src:
    naip_b8 = src.read()

naip_ndvi = es.normalized_diff(naip_b8, naip_b4)

with rio.open(cmh_b8_g1_p21mira) as src:
    naip_data = src.read()
    naip_meta = src.profile

naip_transform = naip_meta["transform"]
naip_crs = naip_meta["crs"]
naip_meta['dtype'] = "float64"

imagem = naip_ndvi.squeeze()
xmin, xmax, ymin, ymax = (0, 35, 0, 60)
imshow_kwargs = {
    'vmax': 1,
    'vmin': -1,
    'cmap': 'RdYlGn',
    'extent': (xmin, xmax, ymin, ymax),
}

plt.imshow(imagem, **imshow_kwargs)
plt.colorbar()
plt.title("NDVI - Gaiola 1 P21 Mirapasto" + " " + "-" + " " + data)
plt.savefig(caminho_imgrgb + '\g1_p21mira.png')
plt.close()

ep.hist(naip_ndvi,
        figsize=(12, 6),
        title=["Histograma - Gaiola 1 P21 Mirapasto" + " " + "-" + " " + data])
plt.savefig(caminho_hist + '\g1_p21mira.png')
plt.close()

media = naip_ndvi.mean()
desvio = naip_ndvi.std()
variancia = naip_ndvi.var()
idpotreiro = 4
idsubarea = 34
area = 'NDVI Gaiola 1 P21 Mirapasto'
id14 = id + 14

query =  "INSERT INTO ndvi (id, idpotreiro, data, ndvi_medio, ndvi_desviop, ndvi_variance, descricao, idsubarea) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
values = (id14, idpotreiro, data, media, desvio, variancia, area, idsubarea)

cur.execute(query, values)
conn.commit()

print("Inseriu")

naip_ndvi_outpath = os.path.join(caminho, 'recorte_ndvi_' + data_img + '_g1_p21mira.tif')
 
# Write your the ndvi raster object
with rio.open(naip_ndvi_outpath, 'w', **naip_meta) as dst:
    dst.write(naip_ndvi)

################################################GAIOLA 1 P21 MIRAPASTO##################################################

################################################GAIOLA 2 P21 MIRAPASTO##################################################

with rio.open(cmh_b4_g2_p21mira) as src:
    naip_b4 = src.read()

with rio.open(cmh_b8_g2_p21mira) as src:
    naip_b8 = src.read()

naip_ndvi = es.normalized_diff(naip_b8, naip_b4)

with rio.open(cmh_b8_g2_p21mira) as src:
    naip_data = src.read()
    naip_meta = src.profile

naip_transform = naip_meta["transform"]
naip_crs = naip_meta["crs"]
naip_meta['dtype'] = "float64"

imagem = naip_ndvi.squeeze()
xmin, xmax, ymin, ymax = (0, 35, 0, 60)
imshow_kwargs = {
    'vmax': 1,
    'vmin': -1,
    'cmap': 'RdYlGn',
    'extent': (xmin, xmax, ymin, ymax),
}

plt.imshow(imagem, **imshow_kwargs)
plt.colorbar()
plt.title("NDVI - Gaiola 2 P21 Mirapasto" + " " + "-" + " " + data)
plt.savefig(caminho_imgrgb + '\g2_p21mira.png')
plt.close()

ep.hist(naip_ndvi,
        figsize=(12, 6),
        title=["Histograma - Gaiola 2 P21 Mirapasto" + " " + "-" + " " + data])
plt.savefig(caminho_hist + '\g2_p21mira.png')
plt.close()

media = naip_ndvi.mean()
desvio = naip_ndvi.std()
variancia = naip_ndvi.var()
idpotreiro = 4
idsubarea = 35
area = 'NDVI Gaiola 2 P21 Mirapasto'
id15 = id + 15

query =  "INSERT INTO ndvi (id, idpotreiro, data, ndvi_medio, ndvi_desviop, ndvi_variance, descricao, idsubarea) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
values = (id15, idpotreiro, data, media, desvio, variancia, area, idsubarea)

cur.execute(query, values)
conn.commit()

print("Inseriu")

naip_ndvi_outpath = os.path.join(caminho, 'recorte_ndvi_' + data_img + '_g2_p21mira.tif')
 
# Write your the ndvi raster object
with rio.open(naip_ndvi_outpath, 'w', **naip_meta) as dst:
    dst.write(naip_ndvi)

################################################GAIOLA 2 P21 MIRAPASTO##################################################

################################################GAIOLA 4 P21 MIRAPASTO##################################################

with rio.open(cmh_b4_g4_p21mira) as src:
    naip_b4 = src.read()

with rio.open(cmh_b8_g4_p21mira) as src:
    naip_b8 = src.read()

naip_ndvi = es.normalized_diff(naip_b8, naip_b4)

with rio.open(cmh_b8_g4_p21mira) as src:
    naip_data = src.read()
    naip_meta = src.profile

naip_transform = naip_meta["transform"]
naip_crs = naip_meta["crs"]
naip_meta['dtype'] = "float64"

imagem = naip_ndvi.squeeze()
xmin, xmax, ymin, ymax = (0, 35, 0, 60)
imshow_kwargs = {
    'vmax': 1,
    'vmin': -1,
    'cmap': 'RdYlGn',
    'extent': (xmin, xmax, ymin, ymax),
}

plt.imshow(imagem, **imshow_kwargs)
plt.colorbar()
plt.title("NDVI - Gaiola 4 P21 Mirapasto" + " " + "-" + " " + data)
plt.savefig(caminho_imgrgb + '\g4_p21mira.png')
plt.close()

ep.hist(naip_ndvi,
        figsize=(12, 6),
        title=["Histograma - Gaiola 4 P21 Mirapasto" + " " + "-" + " " + data])
plt.savefig(caminho_hist + '\g4_p21mira.png')
plt.close()

media = naip_ndvi.mean()
desvio = naip_ndvi.std()
variancia = naip_ndvi.var()
idpotreiro = 4
idsubarea = 36
area = 'NDVI Gaiola 4 P21 Mirapasto'
id16 = id + 16

query =  "INSERT INTO ndvi (id, idpotreiro, data, ndvi_medio, ndvi_desviop, ndvi_variance, descricao, idsubarea) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
values = (id16, idpotreiro, data, media, desvio, variancia, area, idsubarea)

cur.execute(query, values)
conn.commit()

print("Inseriu")

naip_ndvi_outpath = os.path.join(caminho, 'recorte_ndvi_' + data_img + '_g4_p21mira.tif')
 
# Write your the ndvi raster object
with rio.open(naip_ndvi_outpath, 'w', **naip_meta) as dst:
    dst.write(naip_ndvi)

################################################GAIOLA 1 P21 MIRAPASTO##################################################
