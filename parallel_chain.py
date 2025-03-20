from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser   
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv
load_dotenv()

from dotenv import load_dotenv

load_dotenv()
model1=ChatOpenAI()
model2= ChatOpenAI()

prompt1=PromptTemplate(
    template='generate a short and simple note of the followint {text} \n',
    input_variables=['text']
)

prompt2=PromptTemplate(
    template='generate a 5 Question for Machine Learning Algo Examination of the follwing {text} \n',
    input_variables=['text']
)

prompt3=PromptTemplate(
    template='merge the provided notes and 5 Questions into a single documents \n notes: {notes} \n QA: {QA}',
    input_variables=['notes','QA']
)

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'QA': prompt2 | model2 | parser,
})
merged_chain=prompt3 | model1 | parser

chain=parallel_chain | merged_chain 
text="""
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""
result=chain.invoke({'text':text})
print(result)