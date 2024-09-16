import json
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI  # GoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

with open('dados_dataset.json', 'r', encoding='utf-8') as f:
    dados_json = json.load(f)

dados_limitados = dados_json[:100]

contexto    = json.dumps(dados_limitados, ensure_ascii=False, indent=4)

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
    resposta = chain.invoke({
        "context": contexto,
        "input_text": input("Digite a sua pergunta: ")
    })
    
    resposta = parser.invoke(resposta)
    print(resposta)
except Exception as e:
    print(f"Ocorreu um erro: {e}")
