import requests
import os
import wget
import pandas as pd
import json
from pandas.io.json import json_normalize

with open(r'C:\Users\Ânderson Fischoeder\Desktop\Script_DadosClimaticos\dados_janeiro.json', 'r') as f:
    data = json.load(f)
df = json_normalize(data)

print(df.columns)

df.to_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Script_DadosClimaticos\dados_janeiro.csv', index=False)