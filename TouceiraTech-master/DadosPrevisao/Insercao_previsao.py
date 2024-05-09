import requests
import base64
import json
import bs4
import csv
import os
import psycopg2
import pandas as pd
import sys
from datetime import date
from pyeto import fao56_penman_monteith, deg2rad, net_rad, psy_const, delta_svp, celsius2kelvin, wind_speed_2m, et_rad, sol_dec, sunset_hour_angle, inv_rel_dist_earth_sun, cs_rad, sol_rad_from_t, avp_from_rhmean, svp_from_t, delta_svp, avp_from_rhmean, atm_pressure, net_out_lw_rad, psy_const, net_in_sol_rad
from pathlib import Path
import numpy as np
import glob
from datetime import date
from datetime import timedelta

# 3 de abril de 2019
d = date(2019, 10, 23)
# somar 1 dia = 4 de abril de 2019
c = 0
id = 1
try:   
#Conecta com o BD
	conn = psycopg2.connect("host='localhost' port='5432' dbname='db_2019' user='postgres' password='12345'")
	#con = psycopg2.connect("host='localhost' port='5432' dbname='pastagemOutlier' user='postgres' password='123456'")	
	cur = conn.cursor()
except:
	log.write("\nFalha na conexão com o BD!")
	
cur.execute("delete from dados_previsao")
conn.commit()	 

##########################################################Chuva####################################################################
for file_name in glob.glob(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\DadosPrevisao\Pastagem_2019\Chuva\*.csv'):
	x = np.genfromtxt(file_name,delimiter=',')
	if c == 4:
		d = d + timedelta(days = 1)
		c=0
		query =	 "INSERT INTO dados_previsao (chuva, data, id) VALUES (%s, %s, %s)"
		values = (x[6], d, id)
		cur.execute(query, values)
		conn.commit()
	elif c<=3:
		query =	 "INSERT INTO dados_previsao (chuva, data, id) VALUES (%s, %s, %s)"
		values = (x[6], d, id)
		cur.execute(query, values)
		conn.commit()
	c = c + 1
	id = id + 1
##########################################################Componente U Vento####################################################################
id = 1	 
for file_name in glob.glob(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\DadosPrevisao\Pastagem_2019\Comp_U_vento\*.csv'):
	x = np.genfromtxt(file_name,delimiter=',')
	cur.execute("update dados_previsao set comp_u_vento=%s where id= '%s';" % (x[6], id))
	conn.commit()
	id = id + 1
##########################################################Componente V Vento####################################################################   
id = 1	 
for file_name in glob.glob(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\DadosPrevisao\Pastagem_2019\Comp_V_vento\*.csv'):
	x = np.genfromtxt(file_name,delimiter=',')
	cur.execute("update dados_previsao set comp_v_vento=%s where id= '%s';" % (x[6], id))
	conn.commit()
	id = id + 1
##########################################################Radiacao####################################################################	 
id = 1	 
for file_name in glob.glob(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\DadosPrevisao\Pastagem_2019\Radiacao\*.csv'):
	x = np.genfromtxt(file_name,delimiter=',')
	cur.execute("update dados_previsao set radiacao=%s where id= '%s';" % (x[6], id))
	conn.commit()
	id = id + 1
##########################################################Temperatura Média####################################################################	  
id = 1	 
for file_name in glob.glob(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\DadosPrevisao\Pastagem_2019\Temp\*.csv'):
	x = np.genfromtxt(file_name,delimiter=',')
	cur.execute("update dados_previsao set temp_media=%s where id= '%s';" % ((x[6] - 273.15), id))
	conn.commit()
	id = id + 1
##########################################################Temperatura Máxima####################################################################   
id = 1	 
for file_name in glob.glob(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\DadosPrevisao\Pastagem_2019\Tmax\*.csv'):
	x = np.genfromtxt(file_name,delimiter=',')
	cur.execute("update dados_previsao set tmax=%s where id= '%s';" % ((x[6] - 273.15), id))
	conn.commit()
	id = id + 1
##########################################################Temperatura Mínima####################################################################   
id = 1	 
for file_name in glob.glob(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\DadosPrevisao\Pastagem_2019\Tmin\*.csv'):
	x = np.genfromtxt(file_name,delimiter=',')
	cur.execute("update dados_previsao set tmin=%s where id= '%s';" % ((x[6] - 273.15), id))
	conn.commit()
	id = id + 1
##########################################################Umidade####################################################################	
id = 1	 
for file_name in glob.glob(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\DadosPrevisao\Pastagem_2019\Umidade\*.csv'):
	x = np.genfromtxt(file_name,delimiter=',')
	cur.execute("update dados_previsao set umidade=%s where id= '%s';" % (x[6], id))
	conn.commit()
	id = id + 1
