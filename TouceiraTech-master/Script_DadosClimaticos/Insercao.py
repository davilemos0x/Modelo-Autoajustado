import requests
import base64
import json
import bs4
import csv
import os
import psycopg2
import pandas as pd
import sys
from datetime import date, datetime
from pyeto import fao56_penman_monteith, deg2rad, net_rad, psy_const, delta_svp, celsius2kelvin, wind_speed_2m, et_rad, sol_dec, sunset_hour_angle, inv_rel_dist_earth_sun, cs_rad, sol_rad_from_t, avp_from_rhmean, svp_from_t, delta_svp, avp_from_rhmean, atm_pressure, net_out_lw_rad, psy_const, net_in_sol_rad

day=1
anoMes="2013-12"
while (day <= 31):
	'''if day==1:
		day=day+3'''
	
	if day<10:
		day2 = str(day)
		data = anoMes + '-' + '0' + day2
	else:
		day2 = str(day)
		data = anoMes + '-' + day2
		
	#data = "2007-12" + "-" + dia

	temp_med = 0
	temp_max_dia = 0
	temp_min_dia = 50
	chuva = 0
	rad = 0
	umid_med = 0
	med_vel_vento = 0
	temp_basal = 10
	deficit = 0.0
	exc = 2.0
	evapo = 3

	caminho = r'C:\Users\Ânderson Fischoeder\Desktop\Script_DadosClimaticos'

	csv = os.path.join(caminho, 'A827-' + data + '.csv')

	df = pd.read_csv(csv)
	#Temp med
	df_temp_med = df.loc[(df['DT_MEDICAO'] == data)]
	temp_med = df_temp_med['TEM_INS'].mean().astype(float)
	#Temp max dia
	df_temp_max_dia = df.loc[(df['DT_MEDICAO'] == data)]
	temp_max_dia = df_temp_max_dia['TEM_MAX'].max().astype(float)
	#Temp min dia
	df_temp_min_dia = df.loc[(df['DT_MEDICAO'] == data)]
	temp_min_dia = df_temp_min_dia['TEM_MIN'].min().astype(float)
	#Media vel do vento
	df_vel_vento = df.loc[(df['DT_MEDICAO'] == data)]
	med_vel_vento = df_vel_vento['VEN_VEL'].mean().astype(float)
	#Somatorio radiacao
	df_rad = df.loc[(df['DT_MEDICAO'] == data)]
	rad_somatorio = df_rad['RAD_GLO'].sum()/1000
	#Acumulado de chuva
	df_chuva = df.loc[(df['DT_MEDICAO'] == data)]
	chuva = df_chuva['CHUVA'].sum().astype(float)
	#soma termica
	soma_termica = (((temp_max_dia+temp_min_dia)/2) - temp_basal).astype(float)
	#Umidades
	df_umid_min = df.loc[(df['DT_MEDICAO'] == data)]
	umid_min = df_umid_min['UMD_MIN'].min().astype(float)
	df_umid_max = df.loc[(df['DT_MEDICAO'] == data)]
	umid_max = df_umid_max['UMD_MAX'].max().astype(float)
	med_umidade = (umid_max + umid_min) / 2
	df_umid_inst = df.loc[(df['DT_MEDICAO'] == data)]
	umid_med = df_umid_inst['UMD_INS'].mean().astype(float)

	evapo_calc = 0

	###############################################################################

	print(data)
	print("Temperatura media: %f" % temp_med)
	print("Temperatura maxima do dia: %f" % temp_max_dia)
	print("Temperatura minima do dia: %f" % temp_min_dia)
	print("Umidade media: %f" % umid_med)
	print("Media velocidade do vento: %f" % med_vel_vento)
	print("Somatorio radiacao: %f" % rad_somatorio)
	print("Acumulado de chuva: %f" % chuva)
	print("Evapo calculada: %f" % evapo_calc)
	print("Soma termica: %f" % soma_termica)
	print("deficit: %f" % deficit)
	print("excesso: %f" % exc)

	try:   
	#Conecta com o BD
		conn = psycopg2.connect("host='localhost' port='5432' dbname='db_2019' user='postgres' password='12345'")
		#con = psycopg2.connect("host='localhost' port='5432' dbname='pastagemOutlier' user='postgres' password='123456'")	
		cur = conn.cursor()
	except:
		log.write("\nFalha na conexão com o BD!")


	query =	 "INSERT INTO clima_evapo (data, tmin, tmed, tmax, umidade, velocidadevento, radiacaosolar, chuva, evapo, somatermica, def, exc) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;"
	values = (data, temp_min_dia, temp_med, temp_max_dia, umid_med, med_vel_vento, rad_somatorio, chuva, evapo_calc, soma_termica, deficit, exc)

	cur.execute(query, values)
	conn.commit()

	print("Inseriu")

	conn.close()
	print("Conexão encerrada")
	day = day + 1

	os.remove(csv)