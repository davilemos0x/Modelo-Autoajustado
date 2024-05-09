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
from datetime import datetime
import matplotlib.pyplot as plt
import math


now = datetime.now()
log = open('log.txt', 'a')
log.write("------ Excecução iniciada em: " + str(now) + "\n\n")

try:
#Conecta com o BD
	con = psycopg2.connect("host='localhost' port='5432' dbname='db_2019' user='postgres' password='12345'")
	#con = psycopg2.connect("host='localhost' port='5432' dbname='pastagemOutlier' user='postgres' password='123456'")
	cur = con.cursor()
		
except:
	log.write("\nFalha na conexão com o BD!")

'''
cur.execute("delete from clima_previsao")
con.commit()
'''
d = date(2019, 10, 23)
data = (d.isoformat(), d.isoformat())
	
file = open("select_dados_previsao.sql", 'r')
sql = " ".join(file.readlines())


cur.execute(sql, data)
con.commit()

try:
	tdo= cur.fetchall()

	ent = []
	prox=None

except:
	print("Erro - Nao foi possivel acessar os dados no BD")
    
c = 0
chuva = 0
vel_vento = 0
radiacao = 0
tmed = 0
tmax = 0
tmin = 100
umidade = 0
id = 2096


while (c<=14):
    query =	 "INSERT INTO real_previsao (id, data, tmin, tmed, tmax, umidade, velocidadevento, radiacaosolar, chuva, evapo, somatermica, def, exc) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    values = (id, d, tmin, tmed, tmax, umidade, vel_vento, radiacao, chuva, 0, 0, 0, 0)
    cur.execute(query, values)
    con.commit()
    c = c + 1
    id = id + 1
    d = d + timedelta(days = 1)
#################################################################################################################################################
d = date(2019, 10, 23)
id = 2096
for x in tdo:
	chuva = chuva + x[1]
	vetor = ((x[2]*x[2])+(x[3]*x[3]))
	vel_vento = vel_vento + math.sqrt(vetor)
	radiacao = radiacao + x[4]
	tmed = tmed + x[5]
	if tmax > x[6]:
		tmax = tmax
	else:
		tmax = x[6]
	if tmin < x[7]:
		tmin = tmin
	else:
		tmin = x[7]
	umidade = umidade + x[8]
	somatermica = ((tmax - tmin)/2) - 10


print(tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0)
cur.execute("update real_previsao set tmin=%s, tmed=%s, tmax=%s, umidade=%s, velocidadevento=%s, radiacaosolar=%s, chuva=%s, evapo=%s, somatermica=%s, def=%s, exc=%s where id= '%s';" % (tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0, id))
con.commit()
id = id + 1
#################################################################################################################################################
d = d + timedelta(days = 1)
data = (d.isoformat(), d.isoformat())
	
file = open("select_dados_previsao.sql", 'r')
sql = " ".join(file.readlines())


cur.execute(sql, data)
con.commit()

try:
	tdo= cur.fetchall()

	ent = []
	prox=None

except:
	print("Erro - Nao foi possivel acessar os dados no BD")

chuva = 0
vel_vento = 0
radiacao = 0
tmed = 0
tmax = 0
tmin = 100
umidade = 0

   
for x in tdo:
	chuva = chuva + x[1]
	vetor = ((x[2]*x[2])+(x[3]*x[3]))
	vel_vento = vel_vento + math.sqrt(vetor)
	radiacao = radiacao + x[4]
	tmed = tmed + x[5]
	if tmax > x[6]:
		tmax = tmax
	else:
		tmax = x[6]
	if tmin < x[7]:
		tmin = tmin
	else:
		tmin = x[7]
	umidade = umidade + x[8]
	somatermica = ((tmax - tmin)/2) - 10


print(tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0)
cur.execute("update real_previsao set tmin=%s, tmed=%s, tmax=%s, umidade=%s, velocidadevento=%s, radiacaosolar=%s, chuva=%s, evapo=%s, somatermica=%s, def=%s, exc=%s where id= '%s';" % (tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0, id))
con.commit()
id = id + 1
#################################################################################################################################################
d = d + timedelta(days = 1)
data = (d.isoformat(), d.isoformat())
	
file = open("select_dados_previsao.sql", 'r')
sql = " ".join(file.readlines())


cur.execute(sql, data)
con.commit()

try:
	tdo= cur.fetchall()

	ent = []
	prox=None

except:
	print("Erro - Nao foi possivel acessar os dados no BD")

chuva = 0
vel_vento = 0
radiacao = 0
tmed = 0
tmax = 0
tmin = 100
umidade = 0
   
for x in tdo:
	chuva = chuva + x[1]
	vetor = ((x[2]*x[2])+(x[3]*x[3]))
	vel_vento = vel_vento + math.sqrt(vetor)
	radiacao = radiacao + x[4]
	tmed = tmed + x[5]
	if tmax > x[6]:
		tmax = tmax
	else:
		tmax = x[6]
	if tmin < x[7]:
		tmin = tmin
	else:
		tmin = x[7]
	umidade = umidade + x[8]
	somatermica = ((tmax - tmin)/2) - 10
	
print(tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0)
cur.execute("update real_previsao set tmin=%s, tmed=%s, tmax=%s, umidade=%s, velocidadevento=%s, radiacaosolar=%s, chuva=%s, evapo=%s, somatermica=%s, def=%s, exc=%s where id= '%s';" % (tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0, id))
con.commit()		
id = id + 1
#################################################################################################################################################
d = d + timedelta(days = 1)
data = (d.isoformat(), d.isoformat())
	
file = open("select_dados_previsao.sql", 'r')
sql = " ".join(file.readlines())


cur.execute(sql, data)
con.commit()

try:
	tdo= cur.fetchall()

	ent = []
	prox=None

except:
	print("Erro - Nao foi possivel acessar os dados no BD")

chuva = 0
vel_vento = 0
radiacao = 0
tmed = 0
tmax = 0
tmin = 100
umidade = 0
   
for x in tdo:
	chuva = chuva + x[1]
	vetor = ((x[2]*x[2])+(x[3]*x[3]))
	vel_vento = vel_vento + math.sqrt(vetor)
	radiacao = radiacao + x[4]
	tmed = tmed + x[5]
	if tmax > x[6]:
		tmax = tmax
	else:
		tmax = x[6]
	if tmin < x[7]:
		tmin = tmin
	else:
		tmin = x[7]
	umidade = umidade + x[8]
	somatermica = ((tmax - tmin)/2) - 10
	
print(tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0)
cur.execute("update real_previsao set tmin=%s, tmed=%s, tmax=%s, umidade=%s, velocidadevento=%s, radiacaosolar=%s, chuva=%s, evapo=%s, somatermica=%s, def=%s, exc=%s where id= '%s';" % (tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0, id))
con.commit()		
id = id + 1
#################################################################################################################################################
d = d + timedelta(days = 1)
data = (d.isoformat(), d.isoformat())
	
file = open("select_dados_previsao.sql", 'r')
sql = " ".join(file.readlines())


cur.execute(sql, data)
con.commit()

try:
	tdo= cur.fetchall()

	ent = []
	prox=None

except:
	print("Erro - Nao foi possivel acessar os dados no BD")

chuva = 0
vel_vento = 0
radiacao = 0
tmed = 0
tmax = 0
tmin = 100
umidade = 0
   
for x in tdo:
	chuva = chuva + x[1]
	vetor = ((x[2]*x[2])+(x[3]*x[3]))
	vel_vento = vel_vento + math.sqrt(vetor)
	radiacao = radiacao + x[4]
	tmed = tmed + x[5]
	if tmax > x[6]:
		tmax = tmax
	else:
		tmax = x[6]
	if tmin < x[7]:
		tmin = tmin
	else:
		tmin = x[7]
	umidade = umidade + x[8]
	somatermica = ((tmax - tmin)/2) - 10
	
print(tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0)
cur.execute("update real_previsao set tmin=%s, tmed=%s, tmax=%s, umidade=%s, velocidadevento=%s, radiacaosolar=%s, chuva=%s, evapo=%s, somatermica=%s, def=%s, exc=%s where id= '%s';" % (tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0, id))
con.commit()		
id = id + 1
#################################################################################################################################################
d = d + timedelta(days = 1)
data = (d.isoformat(), d.isoformat())
	
file = open("select_dados_previsao.sql", 'r')
sql = " ".join(file.readlines())


cur.execute(sql, data)
con.commit()

try:
	tdo= cur.fetchall()

	ent = []
	prox=None

except:
	print("Erro - Nao foi possivel acessar os dados no BD")

chuva = 0
vel_vento = 0
radiacao = 0
tmed = 0
tmax = 0
tmin = 100
umidade = 0
   
for x in tdo:
	chuva = chuva + x[1]
	vetor = ((x[2]*x[2])+(x[3]*x[3]))
	vel_vento = vel_vento + math.sqrt(vetor)
	radiacao = radiacao + x[4]
	tmed = tmed + x[5]
	if tmax > x[6]:
		tmax = tmax
	else:
		tmax = x[6]
	if tmin < x[7]:
		tmin = tmin
	else:
		tmin = x[7]
	umidade = umidade + x[8]
	somatermica = ((tmax - tmin)/2) - 10
	
print(tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0)
cur.execute("update real_previsao set tmin=%s, tmed=%s, tmax=%s, umidade=%s, velocidadevento=%s, radiacaosolar=%s, chuva=%s, evapo=%s, somatermica=%s, def=%s, exc=%s where id= '%s';" % (tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0, id))
con.commit()		
id = id + 1
#################################################################################################################################################
d = d + timedelta(days = 1)
data = (d.isoformat(), d.isoformat())
	
file = open("select_dados_previsao.sql", 'r')
sql = " ".join(file.readlines())


cur.execute(sql, data)
con.commit()

try:
	tdo= cur.fetchall()

	ent = []
	prox=None

except:
	print("Erro - Nao foi possivel acessar os dados no BD")

chuva = 0
vel_vento = 0
radiacao = 0
tmed = 0
tmax = 0
tmin = 100
umidade = 0
   
for x in tdo:
	chuva = chuva + x[1]
	vetor = ((x[2]*x[2])+(x[3]*x[3]))
	vel_vento = vel_vento + math.sqrt(vetor)
	radiacao = radiacao + x[4]
	tmed = tmed + x[5]
	if tmax > x[6]:
		tmax = tmax
	else:
		tmax = x[6]
	if tmin < x[7]:
		tmin = tmin
	else:
		tmin = x[7]
	umidade = umidade + x[8]
	somatermica = ((tmax - tmin)/2) - 10
	
print(tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0)
cur.execute("update real_previsao set tmin=%s, tmed=%s, tmax=%s, umidade=%s, velocidadevento=%s, radiacaosolar=%s, chuva=%s, evapo=%s, somatermica=%s, def=%s, exc=%s where id= '%s';" % (tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0, id))
con.commit()		
id = id + 1
#################################################################################################################################################
d = d + timedelta(days = 1)
data = (d.isoformat(), d.isoformat())
	
file = open("select_dados_previsao.sql", 'r')
sql = " ".join(file.readlines())


cur.execute(sql, data)
con.commit()

try:
	tdo= cur.fetchall()

	ent = []
	prox=None

except:
	print("Erro - Nao foi possivel acessar os dados no BD")

chuva = 0
vel_vento = 0
radiacao = 0
tmed = 0
tmax = 0
tmin = 100
umidade = 0
   
for x in tdo:
	chuva = chuva + x[1]
	vetor = ((x[2]*x[2])+(x[3]*x[3]))
	vel_vento = vel_vento + math.sqrt(vetor)
	radiacao = radiacao + x[4]
	tmed = tmed + x[5]
	if tmax > x[6]:
		tmax = tmax
	else:
		tmax = x[6]
	if tmin < x[7]:
		tmin = tmin
	else:
		tmin = x[7]
	umidade = umidade + x[8]
	somatermica = ((tmax - tmin)/2) - 10
	
print(tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0)
cur.execute("update real_previsao set tmin=%s, tmed=%s, tmax=%s, umidade=%s, velocidadevento=%s, radiacaosolar=%s, chuva=%s, evapo=%s, somatermica=%s, def=%s, exc=%s where id= '%s';" % (tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0, id))
con.commit()		
id = id + 1
#################################################################################################################################################
d = d + timedelta(days = 1)
data = (d.isoformat(), d.isoformat())
	
file = open("select_dados_previsao.sql", 'r')
sql = " ".join(file.readlines())


cur.execute(sql, data)
con.commit()

try:
	tdo= cur.fetchall()

	ent = []
	prox=None

except:
	print("Erro - Nao foi possivel acessar os dados no BD")

chuva = 0
vel_vento = 0
radiacao = 0
tmed = 0
tmax = 0
tmin = 100
umidade = 0
   
for x in tdo:
	chuva = chuva + x[1]
	vetor = ((x[2]*x[2])+(x[3]*x[3]))
	vel_vento = vel_vento + math.sqrt(vetor)
	radiacao = radiacao + x[4]
	tmed = tmed + x[5]
	if tmax > x[6]:
		tmax = tmax
	else:
		tmax = x[6]
	if tmin < x[7]:
		tmin = tmin
	else:
		tmin = x[7]
	umidade = umidade + x[8]
	somatermica = ((tmax - tmin)/2) - 10
	
print(tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0)
cur.execute("update real_previsao set tmin=%s, tmed=%s, tmax=%s, umidade=%s, velocidadevento=%s, radiacaosolar=%s, chuva=%s, evapo=%s, somatermica=%s, def=%s, exc=%s where id= '%s';" % (tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0, id))
con.commit()		
id = id + 1
#################################################################################################################################################
d = d + timedelta(days = 1)
data = (d.isoformat(), d.isoformat())
	
file = open("select_dados_previsao.sql", 'r')
sql = " ".join(file.readlines())


cur.execute(sql, data)
con.commit()

try:
	tdo= cur.fetchall()

	ent = []
	prox=None

except:
	print("Erro - Nao foi possivel acessar os dados no BD")

chuva = 0
vel_vento = 0
radiacao = 0
tmed = 0
tmax = 0
tmin = 100
umidade = 0
   
for x in tdo:
	chuva = chuva + x[1]
	vetor = ((x[2]*x[2])+(x[3]*x[3]))
	vel_vento = vel_vento + math.sqrt(vetor)
	radiacao = radiacao + x[4]
	tmed = tmed + x[5]
	if tmax > x[6]:
		tmax = tmax
	else:
		tmax = x[6]
	if tmin < x[7]:
		tmin = tmin
	else:
		tmin = x[7]
	umidade = umidade + x[8]
	somatermica = ((tmax - tmin)/2) - 10
	
print(tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0)
cur.execute("update real_previsao set tmin=%s, tmed=%s, tmax=%s, umidade=%s, velocidadevento=%s, radiacaosolar=%s, chuva=%s, evapo=%s, somatermica=%s, def=%s, exc=%s where id= '%s';" % (tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0, id))
con.commit()		
id = id + 1
#################################################################################################################################################
d = d + timedelta(days = 1)
data = (d.isoformat(), d.isoformat())
	
file = open("select_dados_previsao.sql", 'r')
sql = " ".join(file.readlines())


cur.execute(sql, data)
con.commit()

try:
	tdo= cur.fetchall()

	ent = []
	prox=None

except:
	print("Erro - Nao foi possivel acessar os dados no BD")

chuva = 0
vel_vento = 0
radiacao = 0
tmed = 0
tmax = 0
tmin = 100
umidade = 0
   
for x in tdo:
	chuva = chuva + x[1]
	vetor = ((x[2]*x[2])+(x[3]*x[3]))
	vel_vento = vel_vento + math.sqrt(vetor)
	radiacao = radiacao + x[4]
	tmed = tmed + x[5]
	if tmax > x[6]:
		tmax = tmax
	else:
		tmax = x[6]
	if tmin < x[7]:
		tmin = tmin
	else:
		tmin = x[7]
	umidade = umidade + x[8]
	somatermica = ((tmax - tmin)/2) - 10
	
print(tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0)
cur.execute("update real_previsao set tmin=%s, tmed=%s, tmax=%s, umidade=%s, velocidadevento=%s, radiacaosolar=%s, chuva=%s, evapo=%s, somatermica=%s, def=%s, exc=%s where id= '%s';" % (tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0, id))
con.commit()		
id = id + 1
#################################################################################################################################################
d = d + timedelta(days = 1)
data = (d.isoformat(), d.isoformat())
	
file = open("select_dados_previsao.sql", 'r')
sql = " ".join(file.readlines())


cur.execute(sql, data)
con.commit()

try:
	tdo= cur.fetchall()

	ent = []
	prox=None

except:
	print("Erro - Nao foi possivel acessar os dados no BD")

chuva = 0
vel_vento = 0
radiacao = 0
tmed = 0
tmax = 0
tmin = 100
umidade = 0
   
for x in tdo:
	chuva = chuva + x[1]
	vetor = ((x[2]*x[2])+(x[3]*x[3]))
	vel_vento = vel_vento + math.sqrt(vetor)
	radiacao = radiacao + x[4]
	tmed = tmed + x[5]
	if tmax > x[6]:
		tmax = tmax
	else:
		tmax = x[6]
	if tmin < x[7]:
		tmin = tmin
	else:
		tmin = x[7]
	umidade = umidade + x[8]
	somatermica = ((tmax - tmin)/2) - 10
	
print(tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0)
cur.execute("update real_previsao set tmin=%s, tmed=%s, tmax=%s, umidade=%s, velocidadevento=%s, radiacaosolar=%s, chuva=%s, evapo=%s, somatermica=%s, def=%s, exc=%s where id= '%s';" % (tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0, id))
con.commit()		
id = id + 1
#################################################################################################################################################
d = d + timedelta(days = 1)
data = (d.isoformat(), d.isoformat())
	
file = open("select_dados_previsao.sql", 'r')
sql = " ".join(file.readlines())


cur.execute(sql, data)
con.commit()

try:
	tdo= cur.fetchall()

	ent = []
	prox=None

except:
	print("Erro - Nao foi possivel acessar os dados no BD")

chuva = 0
vel_vento = 0
radiacao = 0
tmed = 0
tmax = 0
tmin = 100
umidade = 0
   
for x in tdo:
	chuva = chuva + x[1]
	vetor = ((x[2]*x[2])+(x[3]*x[3]))
	vel_vento = vel_vento + math.sqrt(vetor)
	radiacao = radiacao + x[4]
	tmed = tmed + x[5]
	if tmax > x[6]:
		tmax = tmax
	else:
		tmax = x[6]
	if tmin < x[7]:
		tmin = tmin
	else:
		tmin = x[7]
	umidade = umidade + x[8]
	somatermica = ((tmax - tmin)/2) - 10
	
print(tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0)
cur.execute("update real_previsao set tmin=%s, tmed=%s, tmax=%s, umidade=%s, velocidadevento=%s, radiacaosolar=%s, chuva=%s, evapo=%s, somatermica=%s, def=%s, exc=%s where id= '%s';" % (tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0, id))
con.commit()		
id = id + 1

#################################################################################################################################################
d = d + timedelta(days = 1)
data = (d.isoformat(), d.isoformat())
	
file = open("select_dados_previsao.sql", 'r')
sql = " ".join(file.readlines())


cur.execute(sql, data)
con.commit()

try:
	tdo= cur.fetchall()

	ent = []
	prox=None

except:
	print("Erro - Nao foi possivel acessar os dados no BD")

chuva = 0
vel_vento = 0
radiacao = 0
tmed = 0
tmax = 0
tmin = 100
umidade = 0
   
for x in tdo:
	chuva = chuva + x[1]
	vetor = ((x[2]*x[2])+(x[3]*x[3]))
	vel_vento = vel_vento + math.sqrt(vetor)
	radiacao = radiacao + x[4]
	tmed = tmed + x[5]
	if tmax > x[6]:
		tmax = tmax
	else:
		tmax = x[6]
	if tmin < x[7]:
		tmin = tmin
	else:
		tmin = x[7]
	umidade = umidade + x[8]
	somatermica = ((tmax - tmin)/2) - 10
	
print(tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0)
cur.execute("update real_previsao set tmin=%s, tmed=%s, tmax=%s, umidade=%s, velocidadevento=%s, radiacaosolar=%s, chuva=%s, evapo=%s, somatermica=%s, def=%s, exc=%s where id= '%s';" % (tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0, id))
con.commit()		
id = id + 1
#################################################################################################################################################
d = d + timedelta(days = 1)
data = (d.isoformat(), d.isoformat())
	
file = open("select_dados_previsao.sql", 'r')
sql = " ".join(file.readlines())


cur.execute(sql, data)
con.commit()

try:
	tdo= cur.fetchall()

	ent = []
	prox=None

except:
	print("Erro - Nao foi possivel acessar os dados no BD")

chuva = 0
vel_vento = 0
radiacao = 0
tmed = 0
tmax = 0
tmin = 100
umidade = 0
   
for x in tdo:
	chuva = chuva + x[1]
	vetor = ((x[2]*x[2])+(x[3]*x[3]))
	vel_vento = vel_vento + math.sqrt(vetor)
	radiacao = radiacao + x[4]
	tmed = tmed + x[5]
	if tmax > x[6]:
		tmax = tmax
	else:
		tmax = x[6]
	if tmin < x[7]:
		tmin = tmin
	else:
		tmin = x[7]
	umidade = umidade + x[8]
	somatermica = ((tmax - tmin)/2) - 10
	
print(tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0)
cur.execute("update real_previsao set tmin=%s, tmed=%s, tmax=%s, umidade=%s, velocidadevento=%s, radiacaosolar=%s, chuva=%s, evapo=%s, somatermica=%s, def=%s, exc=%s where id= '%s';" % (tmin, (tmed/4), tmax, (umidade/4), (vel_vento/4), radiacao, chuva, 0, somatermica, 0, 0, id))
con.commit()		
id = id + 1