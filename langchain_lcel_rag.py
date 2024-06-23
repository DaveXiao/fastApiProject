from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_zhipu import ChatPromptTemplate
from zhipuai import ZhipuAI

from config import API_KEY

client = ZhipuAI(api_key=API_KEY)

# 加载文档
loader = PyPDFLoader("llama2.pdf")
pages = loader.load_and_split()

# 文档切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)

texts = text_splitter.create_documents(
    [page.page_content for page in pages[:4]]
)

# 灌库
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
print("type>>>>>>>>>", type(embeddings))
db = FAISS.from_documents(texts, embeddings)

# 检索 top-1 结果
retriever = db.as_retriever(search_kwargs={"k": 5})


# Prompt模板
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


model = ChatOpenAI(
    temperature=0.95,
    model="glm-4",
    openai_api_key=API_KEY,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

# Chain
rag_chain = (
    {"question": RunnablePassthrough(), "context": retriever}
    | prompt
    | model
    | StrOutputParser()
)

rag_chain.invoke("Llama 2有多少参数")