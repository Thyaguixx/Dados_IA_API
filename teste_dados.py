from datasets import load_dataset

dataset = load_dataset("ruanchaves/b2w-reviews01")
dados = dataset['train']

dados_pandas    = dados.to_pandas()
dados_json      = dados_pandas.to_json(orient='records', indent=4, force_ascii=False)

with open('dados_dataset.json', 'w', encoding='utf-8') as f:
    f.write(dados_json)