# Instalar Ambiente Conda (Ambientes)
conda env create -n path/p35.yml
Ou
conda env create -n path/base.yml

Após isso, ativar o ambiente com:
conda activate env.yml (Onde env.yml é o arquivo no qual foi utilizado no ambiente)

# Restaurar base de dados (DataBases)
Através dos arquivos .backup pode ser feito o restore no banco, tanto através do pgadmin quanto por linha de comando. 

## IMPORTANTE
    A extensão POSTGIS deve estar instalada para que o restore do banco funcione corretamente.

# Insercao de planilhas de dados de pastagem (Dados_Naylor e Dados_Cristina)
Com o uso do script "HW.py" é possível ler as planilhas de dados e armazenar no banco de dados.

# Coleta de dados meteorológicos (Script_DadosClimaticos)
Através do script "download.py" informando a data desejada para coleta. Através desse script o arquivo json já é transformado para csv.
Com o script "Insercao.py" os arquivos coletados são então armazenados no banco de dados. Os scripts "evapo.py" e "bh.py" são responsáveis
por fazer a atualização nos valores de ETo, excesso e déficit.

# Coleta de imagens (img_sentinel)
O script "\donwload\sentinel.py" é responsável por realizar a coleta de imagens na data de interesse. Importante atualizar no script as informações
de usuário.

# Manipulação de imagens (img_sentinel)
Através do script "\crop_e_ndvi\corte.py" é possível realizar o recorte das imagens. Importante sempre atualizar os endereços onde estão localizadas as imagens e onde
deseja-se salvar os resultados.
Com o script "\crop_e_ndvi\naip.py" é possível calcular o ndvi através das imagens recortadas. Sempre atualizando os endereços onde estão as imagens.

# Coleta de dados de previsão meteorológica
Através do comando "python rdams-cliente.py -submit ds084.1". Lembrando que é necessário criar um usuário no repositório rda e atualizar o arquivo "rdamspw.txt".
Importante analisar o dataset "084.1" no repositório para verificar o nome das variáveis climáticas que devem ser apresentadas no arquivo "ds084.1". Para cada uma
das variáveis deve ser feita uma requisição, visto que alguns parâmetros são diferentes entre elas.
Os scripts "Insercao_previsao.py" e "Manipulacao_dados_previsao" foram criados no intuito de manter a consistência nos dados coletados e os presentes no banco de dados,
respeitando a ordem de execução. Os scripts "evapo.py" e bh.py" seguem a mesma lógica para os dados meteorológicos reais.

# Modelo de predição
Através do script "rede_mstotal_uniao.py" é possível realizar a predição com o os treinamentos estratificado por tratamento e não estratificado e com o script
"rede_neural_pastagem_instantanea.py" é possível realizar a predição instantanea para o conjunto de dados A (dados do Naylor).
Utilizando o script "rede_mstotal_uniao_cristina.py" e "rede_neural_pastagem_instantanea_cristina.py" segue a mesma lógica. porém, para o conjunto de dados B (dados da Cristina).
Lembrando que é necessário a criação dos arquivos de entrada, obtendo-os através do script "consolida_dados.py" para o conjunto de dados A e "consolida_dados_cristina.py"
para o conjunto de dados B. Feita a criação dos arquivos de entrada para cada potreiro utiliza-se o script "Inserir.py" que faz a inserção dos dados no banco de dados e com
a utilização do script "consolida_dados_uniao.py" é possível gerar as entradas estratificadas por tratamento ou não estratificadas.
Os scripts "Treino_potreiros.py" realiza a predição da mesma forma, porém, é gerado apenas um gráfico de saída para os três tipos de treinamento. O script "Treino_potreiros_cristina.py"
Segue a mesma lógica para os dados do conjunto B.

# Importante
Sempre analisar os endereços onde estão os arquivos e fazer a substituição e também configurar os scripts que necessitam de dados do usuário.

