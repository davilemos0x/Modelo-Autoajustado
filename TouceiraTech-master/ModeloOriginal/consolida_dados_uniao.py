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

file = open("selectent_uniao.sql", 'r')
sql = " ".join(file.readlines())

cur.execute(sql)
con.commit()

try:
	tdo= cur.fetchall()

	ent = []

except:
	print("Erro - Nao foi possivel acessar os dados no BD")

	
for x in tdo:
	ent.append([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19], x[20], x[21], x[22], x[23], x[24],
			   x[25], x[26], x[27], x[28], x[29], x[30], x[31], x[32], x[33], x[34]])
		
	
import csv

#presente
#header = ['data', 'numerodias', 'alturamedia', 'alturamediaanterior', 'pcanonni', 'mstotalanterior', 
#		  'tmin', 'dptmin', 'vartmin', 'tmed', 'dptmed', 'vartmed', 'tmax', 'dptmax', 'vartmax', 
#		  'umidade', 'dpumidade', 'varumidade', 'velocidadevento', 'dpvelocidadevento', 'varvelovidadevento', 
#		  'radiacaosolar', 'dpradiacaosolar', 'varradiacaosolar', 'chuva', 'dpchuva', 'varchuva', 'somatermica', 
#		  'dpsomatermica', 'varsomatermica', 'def', 'dpdef', 'vardef', 'exc', 'dpexc', 'varexc', 'taxaacumulo']

#futuro
header = ['data', 'numerodias', 'alturamediaanterior', 'mstotalanterior', 'tmin', 'dptmin', 'vartmin', 'tmed', 
		   'dptmed', 'vartmed', 'tmax', 'dptmax', 'vartmax', 'umidade', 'dpumidade', 'varumidade', 'velocidadevento', 
		   'dpvelocidadevento', 'varvelovidadevento', 'radiacaosolar', 'dpradiacaosolar', 'varradiacaosolar', 'chuva', 
		   'dpchuva', 'varchuva', 'somatermica', 'dpsomatermica', 'varsomatermica', 'def', 'dpdef', 'vardef', 'exc', 
		   'dpexc', 'varexc', 'taxaacumulo']


#REVERSE
#ent.reverse()

with open('entrada_Inf.csv', 'w', newline='') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,	delimiter=';')
	wr.writerow(header)
	wr.writerows(ent)
