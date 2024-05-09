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


file = open("selectlotacao_cristina_p1.sql", 'r')
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
	if str(x[0])=='2007-06-21' or str(x[0])=='2012-08-14':
		continue
	else:
		ent.append([x[0], '%.2f' % (x[1]/6)])

header = ['data', 'CA']

with open('lotacao_cristina_p1.csv', 'w', newline='') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,	delimiter=';')
	wr.writerow(header)
	wr.writerows(ent)
	
file = open("selectlotacao_cristina_p2.sql", 'r')
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
	if str(x[0])=='2007-06-21' or str(x[0])=='2012-08-14':
		continue
	else:
		ent.append([x[0], '%.2f' % (x[1]/6)])
header = ['data', 'CA']

with open('lotacao_cristina_p2.csv', 'w', newline='') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,	delimiter=';')
	wr.writerow(header)
	wr.writerows(ent)
	
file = open("selectlotacao_cristina_p3.sql", 'r')
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
	if str(x[0])=='2007-06-21' or str(x[0])=='2012-08-14':
		continue
	else:
		ent.append([x[0], '%.2f' % (x[1]/6)])
header = ['data', 'CA']

with open('lotacao_cristina_p3.csv', 'w', newline='') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,	delimiter=';')
	wr.writerow(header)
	wr.writerows(ent)
	
file = open("selectlotacao_cristina_p4.sql", 'r')
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
	if str(x[0])=='2007-06-21' or str(x[0])=='2012-08-14':
		continue
	else:
		ent.append([x[0], '%.2f' % (x[1]/6)])
header = ['data', 'CA']

with open('lotacao_cristina_p4.csv', 'w', newline='') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,	delimiter=';')
	wr.writerow(header)
	wr.writerows(ent)
	
file = open("selectlotacao_cristina_p5.sql", 'r')
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
	if str(x[0])=='2007-06-21' or str(x[0])=='2012-08-14':
		continue
	else:
		ent.append([x[0], '%.2f' % (x[1]/6)])
header = ['data', 'CA']

with open('lotacao_cristina_p5.csv', 'w', newline='') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,	delimiter=';')
	wr.writerow(header)
	wr.writerows(ent)
	
file = open("selectlotacao_cristina_p6.sql", 'r')
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
	if str(x[0])=='2007-06-21' or str(x[0])=='2012-08-14':
		continue
	else:
		ent.append([x[0], '%.2f' % (x[1]/6)])
header = ['data', 'CA']

with open('lotacao_cristina_p6.csv', 'w', newline='') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,	delimiter=';')
	wr.writerow(header)
	wr.writerows(ent)

file = open("selectlotacao_cristina_p7.sql", 'r')
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
	if str(x[0])=='2007-06-21' or str(x[0])=='2012-08-14':
		continue
	else:
		ent.append([x[0], '%.2f' % (x[1]/6)])
header = ['data', 'CA']

with open('lotacao_cristina_p7.csv', 'w', newline='') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,	delimiter=';')
	wr.writerow(header)
	wr.writerows(ent)

file = open("selectlotacao_cristina_p8.sql", 'r')
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
	if str(x[0])=='2007-06-21' or str(x[0])=='2012-08-14':
		continue
	else:
		ent.append([x[0], '%.2f' % (x[1]/6)])
header = ['data', 'CA']

with open('lotacao_cristina_p8.csv', 'w', newline='') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,	delimiter=';')
	wr.writerow(header)
	wr.writerows(ent)

file = open("selectlotacao_cristina_p9.sql", 'r')
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
	if str(x[0])=='2007-06-21' or str(x[0])=='2012-08-14':
		continue
	else:
		ent.append([x[0], '%.2f' % (x[1]/6)])
header = ['data', 'CA']

with open('lotacao_cristina_p9.csv', 'w', newline='') as myfile:
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,	delimiter=';')
	wr.writerow(header)
	wr.writerows(ent)