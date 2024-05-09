import psycopg2
import numpy as np
from datetime import datetime
from datetime import date
import csv
import os
import pandas as pd

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


file = open("selectlotacao1.sql", 'r')
sql = " ".join(file.readlines())

cur.execute(sql)
con.commit()

try:
	tdo= cur.fetchall()
	
	ent = []
	ant=None
	
except:
	print("Erro - Nao foi possivel acessar os dados no BD")

linhas = sum(1 for line in tdo)

for x in tdo:
	if (ant == None): 
		ant=x
		peso = x[1]
		c = 1
		continue
		
	if x[1] == None:
		x = [x[0], 0]
	if x[0] == ant[0]:
		peso = x[1] + peso
		data = x[0]
		if c == (linhas-1):
			peso = ("%.2f" % (peso/7.7))
			ent.append([data, peso])

	else:
		peso = ("%.2f" % (peso/7.7))
		ent.append([data, peso])
		peso = x[1]
	c = c + 1
	ant=x
header = ['data', 'pesoorigem']

with open('lotacao1.csv', 'w', newline='') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,	delimiter=';')
	wr.writerow(header)
	wr.writerows(ent)

file = open("selectlotacao2.sql", 'r')
sql = " ".join(file.readlines())

cur.execute(sql)
con.commit()

try:
	tdo= cur.fetchall()
	
	ent = []
	ant=None
	
except:
	print("Erro - Nao foi possivel acessar os dados no BD")

linhas = sum(1 for line in tdo)


for x in tdo:
	if (ant == None): 
		ant=x
		peso = x[1]
		c = 1
		continue
		
	if x[1] == None:
		x = [x[0], 0]
	if x[0] == ant[0]:
		peso = x[1] + peso
		data = x[0]
		if c == (linhas-1):
			peso = ("%.2f" % (peso/6))
			ent.append([data, peso])
	else:
		peso = ("%.2f" % (peso/6))
		ent.append([data, peso])	
		peso = x[1]
	c = c + 1
	ant=x
header = ['data', 'pesoorigem']

with open('lotacao2.csv', 'w', newline='') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,	delimiter=';')
	wr.writerow(header)
	wr.writerows(ent)


file = open("selectlotacao3.sql", 'r')
sql = " ".join(file.readlines())

cur.execute(sql)
con.commit()

try:
	tdo= cur.fetchall()
	
	ent = []
	ant=None
	
except:
	print("Erro - Nao foi possivel acessar os dados no BD")

linhas = sum(1 for line in tdo)


for x in tdo:
	if (ant == None): 
		ant=x
		peso = x[1]
		c = 1
		continue
		
	if x[1] == None:
		x = [x[0], 0]
	if x[0] == ant[0]:
		peso = x[1] + peso
		data = x[0]
		if c == (linhas-1):
			peso = ("%.2f" % (peso/10))
			ent.append([data, peso])	
	else:
		peso = ("%.2f" % (peso/10))
		ent.append([data, peso])
		peso = x[1]
	c = c + 1
	ant=x
header = ['data', 'pesoorigem']

with open('lotacao3.csv', 'w', newline='') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,	delimiter=';')
	wr.writerow(header)
	wr.writerows(ent)
    
file = open("selectlotacao4.sql", 'r')
sql = " ".join(file.readlines())

cur.execute(sql)
con.commit()

try:
	tdo= cur.fetchall()
	
	ent = []
	ant=None
	
except:
	print("Erro - Nao foi possivel acessar os dados no BD")

linhas = sum(1 for line in tdo)

for x in tdo:
	if (ant == None): 
		ant=x
		peso = x[1]
		c = 1
		continue
		
	if x[1] == None:
		x = [x[0], 0]
	if x[0] == ant[0]:
		peso = x[1] + peso
		data = x[0]
		if c == (linhas-1):
			peso = ("%.2f" % (peso/8.8))
			ent.append([data, peso])	
	else:
		peso = ("%.2f" % (peso/8.8))
		ent.append([data, peso])	
		peso = x[1]
	c = c + 1
	ant=x
header = ['data', 'pesoorigem']

with open('lotacao4.csv', 'w', newline='') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,	delimiter=';')
	wr.writerow(header)
	wr.writerows(ent)