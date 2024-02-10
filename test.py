import os

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def get_database_info():
    return """
    A harmonia musical é a combinação simultânea de notas e a estrutura subjacente de acordes na música. 
    Ela cria profundidade e complexidade, dando emoção à melodia. A harmonia usa escalas, acordes e progressões de acordes para estabelecer o contexto tonal. 
    Ela pode ser consonante (agradável e resoluta) ou dissonante (tensa e instável), criando diferentes emoções. 
    Entender a harmonia é fundamental para a composição e performance musical.
    """


def get_summary_prompt_template():
    template = """
    Você é um músico experiente que fala português do Brasil, mas também sabe ler e escrever em inglês. 
    Você é muito bom em teoria e harmonia musical. 
    Responda baseado nas seguinte informação {database_info}. 
    
    1 - O que é usado pela harmonina para estabelecer o contexto tonal?
    
    """

    return PromptTemplate(id="music_teacher_app",
                          name="Professor de Música",
                          description="Analisa perguntas sobre música e retorna uma interpretação detalhada.",
                          input_variables=["database_info"],
                          template=template)


def get_llm():
    return ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)


def get_chain():
    return LLMChain(llm=get_llm(), prompt=get_summary_prompt_template())


if __name__ == "__main__":
    # Load env variables
    load_dotenv()

    # Get chain
    chain = get_chain()

    # Run
    res = chain.invoke(input={"database_info": get_database_info()})
    print(res['text'])
