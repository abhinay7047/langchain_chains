from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser 
load_dotenv()

prompt = PromptTemplate(
    template='give the imformation about the given {topic}',
    input_variables=['topic']
)

model=ChatOpenAI()
parser=StrOutputParser()

#langchain expressive language
chain= prompt | model | parser
result=chain.invoke({'topic':'Virat Kohli'})

print(result) 

chain.get_graph().print_ascii()