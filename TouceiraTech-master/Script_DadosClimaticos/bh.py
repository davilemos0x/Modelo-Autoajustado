import psycopg2
import numpy as np
from datetime import datetime
import math

now = datetime.now()

log = open('log.txt', 'a')
log.write("------ Excecução iniciada em: " + str(now) + "\n\n")

try:   
#Conecta com o BD
    conn = psycopg2.connect("host='localhost' port='5432' dbname='db_2019' user='postgres' password='12345'")
	#con = psycopg2.connect("host='localhost' port='5432' dbname='pastagemOutlier' user='postgres' password='123456'")  
    cur = conn.cursor()
except:
	log.write("\nFalha na conexão com o BD!")

file = open("selectbalanco.sql", 'r')
sql = " ".join(file.readlines())

cur.execute(sql)

tdo= cur.fetchall()


cad=60;
negacum=0;
arm=60;

for x in tdo:

	armant=arm
	
	p=x[1]
	if x[1] is None:
		p=0
	
	et=x[2]
	if x[2] is None:
		et=0

	pe=p-et
	
	if pe < 0:
		negacum=negacum+pe 
		arm=cad*math.exp(negacum/cad)

	if pe > 0:
		arm=arm+pe
		negacum=cad*math.log(arm/cad)

	if negacum > 0:
		negacum = 0

	if arm > cad:
		arm=cad

	alt=arm-armant

	if pe >= 0:
		etr=et

	if alt<0:
		etr=p+abs(alt)

	DEF=et-etr

	if arm<cad:
		EXC=0

	if arm==cad:
		EXC=pe-alt

	print(x[0],x[1],x[2],pe,negacum,arm, alt, etr, DEF, EXC)
	cur.execute("update clima_evapo set def=%s, exc=%s where data= '%s';" % (round(DEF,2), round(EXC,2), x[0].isoformat()))

conn.commit()
