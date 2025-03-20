from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv  
load_dotenv()

prompt1=PromptTemplate(
    template='generate a detail report {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='generate a 5 pointer summary report \n {text}',
    input_variables=['text']
)

model=ChatOpenAI()
parser=StrOutputParser()

chain =prompt1 | model | parser | prompt2 | model | parser

result =chain.invoke({'topic':'Unemployment in India'})
print(result)
chain.get_graph().print_ascii()

