import psycopg2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

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

file = open("selectentndvi.sql", 'r')
sql = " ".join(file.readlines())

cur.execute(sql)
con.commit()

try:
	tdo= cur.fetchall()

	ent = []
	ant=None

except:
	print("Erro - Nao foi possivel acessar os dados no BD")

	
for x in tdo:  
	if (ant == None): 
		ant=x
		continue
    
	if (ant[0] != x[0]):
		c=x[0]-ant[0]

		file = open("selectclima.sql", 'r')
		sqlclima = " ".join(file.readlines())
        
		data = (ant[0].isoformat(), x[0].isoformat())

		cur.execute(sqlclima, data)
		con.commit()

		clima= cur.fetchall()
		
        #presente
		#ent.append([x[1], (x[1]-ant[1]).days, x[2], ant[2], x[4], ant[3], clima[0][0], clima[0][7],
		#		   clima[0][14], clima[0][1], clima[0][8], clima[0][15], clima[0][2], clima[0][9], 
		#		   clima[0][16], clima[0][3], clima[0][10], clima[0][17], clima[0][4], clima[0][11], 
		#		   clima[0][18], clima[0][5], clima[0][12], clima[0][19], clima[0][6], clima[0][13], 
		#		   clima[0][20], clima[0][21], clima[0][22], clima[0][23],clima[0][24], clima[0][25], 
		#		   clima[0][26],clima[0][27], clima[0][28], clima[0][29], x[3]])
        #futuro
		ent.append([x[0], (x[0]-ant[0]).days, ant[1], clima[0][0], clima[0][7], clima[0][14],
					clima[0][1], clima[0][8], clima[0][15], clima[0][2], clima[0][9], clima[0][16], 
					clima[0][3], clima[0][10], clima[0][17], clima[0][4], clima[0][11], clima[0][18], 
					clima[0][5], clima[0][12], clima[0][19], clima[0][6], clima[0][13], clima[0][20], 
					clima[0][21], clima[0][22], clima[0][23], clima[0][24], clima[0][25], clima[0][26],
					clima[0][27], clima[0][28], clima[0][29], x[1]])
		
	ant=x

import csv

#presente
#header = ['data', 'numerodias', 'alturamedia', 'alturamediaanterior', 'pcanonni', 'mstotalanterior', 
#		  'tmin', 'dptmin', 'vartmin', 'tmed', 'dptmed', 'vartmed', 'tmax', 'dptmax', 'vartmax', 
#		  'umidade', 'dpumidade', 'varumidade', 'velocidadevento', 'dpvelocidadevento', 'varvelovidadevento', 
#		  'radiacaosolar', 'dpradiacaosolar', 'varradiacaosolar', 'chuva', 'dpchuva', 'varchuva', 'somatermica', 
#		  'dpsomatermica', 'varsomatermica', 'def', 'dpdef', 'vardef', 'exc', 'dpexc', 'varexc', 'taxaacumulo']

#futuro
header = ['data', 'numerodias', 'ndvianterior', 'tmin', 'dptmin', 'vartmin', 'tmed', 
		   'dptmed', 'vartmed', 'tmax', 'dptmax', 'vartmax', 'umidade', 'dpumidade', 'varumidade', 'velocidadevento', 
		   'dpvelocidadevento', 'varvelovidadevento', 'radiacaosolar', 'dpradiacaosolar', 'varradiacaosolar', 'chuva', 
		   'dpchuva', 'varchuva', 'somatermica', 'dpsomatermica', 'varsomatermica', 'def', 'dpdef', 'vardef', 'exc', 
		   'dpexc', 'varexc', 'ndvi']


#REVERSE
#ent.reverse()

with open('e2_ndvi_data.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,  delimiter=';')
    wr.writerow(header)
    wr.writerows(ent)
