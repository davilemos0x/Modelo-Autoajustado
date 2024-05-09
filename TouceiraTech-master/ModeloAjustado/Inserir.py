import csv
import os
import psycopg2
import pandas as pd
import sys
from datetime import date
import numpy as np

try:   
#Conecta com o BD
	conn = psycopg2.connect("host='localhost' port='5432' dbname='db_2019' user='postgres' password='12345'")
	#con = psycopg2.connect("host='localhost' port='5432' dbname='pastagemOutlier' user='postgres' password='123456'")	
	cur = conn.cursor()
except:
	log.write("\nFalha na conexão com o BD!")

cur.execute('DELETE FROM pastagens_ca')
conn.commit()

with open(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\ModeloAjustado\e1_cristina_Treino.csv', 'r') as f:
    # Notice that we don't need the `csv` module.
    next(f) # Skip the header row.
    cur.copy_from(f, 'pastagens_ca', sep=';')
conn.commit()

with open(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\ModeloAjustado\e2_cristina_Treino.csv', 'r') as f:
    # Notice that we don't need the `csv` module.
    next(f) # Skip the header row.
    cur.copy_from(f, 'pastagens_ca', sep=';')
conn.commit()

with open(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\ModeloAjustado\e7_cristina_Treino.csv', 'r') as f:
    # Notice that we don't need the `csv` module.
    next(f) # Skip the header row.
    cur.copy_from(f, 'pastagens_ca', sep=';')
conn.commit()

with open(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\ModeloAjustado\e2_Treino.csv', 'r') as f:
    # Notice that we don't need the `csv` module.
    next(f) # Skip the header row.
    cur.copy_from(f, 'pastagens_ca', sep=';')
conn.commit()

with open(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\ModeloAjustado\e4_Treino.csv', 'r') as f:
    # Notice that we don't need the `csv` module.
    next(f) # Skip the header row.
    cur.copy_from(f, 'pastagens_ca', sep=';')
conn.commit()

with open(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\ModeloAjustado\e4_cristina_Treino.csv', 'r') as f:
    # Notice that we don't need the `csv` module.
    next(f) # Skip the header row.
    cur.copy_from(f, 'pastagens_ca', sep=';')
conn.commit()

with open(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\ModeloAjustado\e5_cristina_Treino.csv', 'r') as f:
    # Notice that we don't need the `csv` module.
    next(f) # Skip the header row.
    cur.copy_from(f, 'pastagens_ca', sep=';')
conn.commit()

with open(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\ModeloAjustado\e9_cristina_Treino.csv', 'r') as f:
    # Notice that we don't need the `csv` module.
    next(f) # Skip the header row.
    cur.copy_from(f, 'pastagens_ca', sep=';')
conn.commit()