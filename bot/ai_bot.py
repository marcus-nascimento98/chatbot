import os

from decouple import config

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

os.environ['GROQ_API_KEY'] = config('GROQ_API_KEY')

class AIBot:

    def __init__(self):
        self.__chat = ChatGroq(model='llama-3.1-70b-versatile')

    def invoke(self, question):
        prompt = PromptTemplate(
            input_variables=['texto'],
            template='''
            Você é um chatbot desenvolvido para ajudar a empresa Allpha a converter vendas e fornecer informações claras e úteis sobre seus produtos. O carro-chefe da empresa é um suplemento 3 em 1, que combina creatina, beta alanina e colágeno hidrolisado. Seu principal público-alvo são atletas e pessoas que buscam melhorar seu desempenho físico e cuidar da saúde. Seu tom deve ser amigável, informativo e focado em destacar os benefícios do produto de forma atrativa e acessível. Eu quero que você seja simples, bem humana, use palavras claras e não escreva muito. Eu percebi que toda hora você repete a mesma coisa no começo, isso faz com que a conversa não seja nada natural. Tente não repetir por exemplo "Olá, estou aqui para te ajudar".
            <texto>
            {texto}
            </texto>
            '''
        )
        chain = prompt | self.__chat | StrOutputParser()
        response = chain.invoke({
            'texto': question,
        })
        return response