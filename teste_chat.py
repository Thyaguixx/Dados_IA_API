import json
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI  # GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# with open('dados_dataset.json', 'r', encoding='utf-8') as f:
#     dados_json = json.load(f)

dados_json      = pd.read_json("dados_dataset.json")
dados_json      = dados_json.drop_duplicates(subset=['reviewer_id'])
dados_limitados = dados_json[:100]

contexto        = dados_limitados.to_dict(orient="records")
contexto_json   = json.dumps(contexto, ensure_ascii=False, indent=4)

llm         = ChatGoogleGenerativeAI(model="gemini-pro")
parser      = StrOutputParser()

#TEMPLATE
prompt = ChatPromptTemplate.from_template("""
Responda as perguntas com base somente nestes dados: {context}
Pergunta: {input_text}                                       
""")

chain = prompt | llm

# Invocar o LLM com o contexto do JSON e a pergunta do usuário
try:
    while True:
        pergunta_usuario = input("Digite a sua pergunta (ou digite SAIR para parar): ")
        
        if pergunta_usuario.strip().upper() == 'SAIR':
            print("Fechando o chat")
            break
        
        resposta = chain.invoke({
            "context": contexto_json,
            "input_text": pergunta_usuario
        })
        
        resposta = parser.invoke(resposta)
        print(resposta)
except Exception as e:
    print(f"Ocorreu um erro: {e}")
