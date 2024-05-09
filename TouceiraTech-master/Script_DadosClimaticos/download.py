import requests
import os
import wget
import pandas as pd
import json
from pandas.io.json import json_normalize

day=10
anoMes="2021-01"
while (day <= 31):	
	if day<10:
		day2 = str(day)
		d = anoMes + '-' + '0' + day2
	else:
		day2 = str(day)
		d = anoMes + '-' + day2
	station = 'A827'

	def baixar_arquivo(url, endereco=None):
		if endereco is None:
			endereco = os.path.basename(url.split("?")[0])
		resposta = requests.get(url)
		if resposta.status_code == requests.codes.OK:
			nome = station + "-" + d + ".json"
			with open(nome, 'wb') as novo_arquivo:
					novo_arquivo.write(resposta.content)
			#print("Download finalizado. Arquivo salvo em: {}".format(endereco))
		else:
			resposta.raise_for_status()


	if __name__ == "__main__":
		# testando a função	   
		test_url = "https://apitempo.inmet.gov.br/estacao/" + d + "/" + d + "/" + station
		baixar_arquivo(test_url)

	caminho = r'C:\Users\Ânderson Fischoeder\Desktop\Script_DadosClimaticos'
	arq = os.path.join(caminho, station + '-' + d + '.json')
	
	with open(station + '-' + d + '.json', 'r') as f:
		data = json.load(f)
	df = json_normalize(data)

	#print(df.columns)
	#print(d)
	#caminho = r'C:\Users\Ânderson Fischoeder\Desktop\Script_DadosClimaticos'
	#cmh = os.join(caminho, 'A827' + '- '+ ' '.join(data) + '.csv')
	csv = 'A827' + '-' + str(d) + '.csv'

	df.to_csv(csv, index=False)
	day = day + 1
	os.remove(arq)