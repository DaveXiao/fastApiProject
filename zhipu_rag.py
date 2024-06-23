from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
# from zhipuai import ZhipuAI
from config import API_KEY
from openai import OpenAI
import numpy as np
from numpy import dot
from numpy.linalg import norm
import chromadb
from chromadb.config import Settings
from nltk.tokenize import sent_tokenize
import nltk
import json


# nltk.download('punkt')


def extract_text_from_pdf(filename, page_numbers=None, min_line_length=1):
    '''从 PDF 文件中（按指定页码）提取文字'''
    paragraphs = []
    buffer = ''
    full_text = ''
    # 提取全部文本
    for i, page_layout in enumerate(extract_pages(filename)):
        # 如果指定了页码范围，跳过范围外的页
        if page_numbers is not None and i not in page_numbers:
            continue
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'
    # 按空行分隔，将文本重新组织成段落
    lines = full_text.split('\n')
    for text in lines:
        if len(text) >= min_line_length:
            buffer += (' ' + text) if not text.endswith('-') else text.strip('-')
        elif buffer:
            paragraphs.append(buffer)
            buffer = ''
    if buffer:
        paragraphs.append(buffer)
    return paragraphs


# paragraphs = extract_text_from_pdf("llama2.pdf", min_line_length=10)
paragraphs = extract_text_from_pdf("llama2.pdf", page_numbers=[2, 3], min_line_length=10)

# for para in paragraphs[:4]:
#     print(para + "\n")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    default_headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
)


def get_completion(prompt, model="glm-4"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
    )
    return response.choices[0].message.content


# 创建build_prompt函数组装prompt模板
def build_prompt(prompt, **kwargs):
    """
    根据给定的模板(prompt)及关键字参数(kwargs)，构建完整的提示信息。

    此函数允许传入额外的参数来定制化prompt内容。如果参数值为字符串列表，这些字符串将被换行符连接。
    其他类型的参数值将直接插入到模板中。

    Args:
        prompt (str): 提示模板字符串，可以包含格式化字段，如 {key}。
        **kwargs: 变长关键字参数，用于填充模板中的字段。
                  如果值为字符串列表，则列表中的每个字符串元素将以换行形式拼接。

    Returns:
        str: 完成构建的提示字符串，所有kwargs中的键值对已被格式化插入到prompt中。
    """
    # 初始化一个字典来存储处理后的输入值
    inputs = {}

    # 遍历传入的关键字参数
    for key, value in kwargs.items():
        # 检查值是否为字符串列表且列表中所有元素都是字符串
        if isinstance(value, list) and all(isinstance(elem, str) for elem in value):
            # 如果是，将列表中的字符串以换行符连接
            val = '\n\n'.join(value)
        else:
            # 否则，直接使用该值
            val = value

        # 将处理后的值存入inputs字典
        inputs[key] = val

    # 使用format方法将inputs字典中的值填充到prompt模板中，并返回结果
    return prompt.format(**inputs)


prompt_template = """
你是一个问答机器人。
你的任务是根据下述给定的已知信息回答用户问题。

已知信息:
{context}

用户问：
{query}

如果已知信息不包含用户问题的答案，或者已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。
请不要输出已知信息中不包含的信息或答案。
请用中文回答用户问题
"""


def cos_sim(a, b):
    '''余弦距离 -- 越大越相似'''
    return dot(a, b) / (norm(a) * norm(b))


def l2(a, b):
    '''欧氏距离 -- 越小越相似'''
    x = np.asarray(a) - np.asarray(b)
    return norm(x)


def get_embeddings(texts, model="embedding-2", dimensions=None):
    '''封装 zhipuAI 的 Embedding 模型接口'''
    if model == "embedding-2":
        dimensions = None
    if dimensions:
        data = client.embeddings.create(
            input=texts, model=model, dimensions=dimensions).data
    else:
        data = client.embeddings.create(input=texts, model=model).data
    return [x.embedding for x in data]


class MyVectorDBConnector:
    def __init__(self, collection_name, embedding_fn):
        chroma_client = chromadb.Client(Settings(allow_reset=True))

        # 为了演示，实际不需要每次 reset()
        chroma_client.reset()

        # 创建一个 collection
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name)
        self.embedding_fn = embedding_fn

    def add_documents(self, documents):
        '''向 collection 中添加文档与向量'''
        self.collection.add(
            embeddings=self.embedding_fn(documents),  # 每个文档的向量
            documents=documents,  # 文档的原文
            ids=[f"id{i}" for i in range(len(documents))]  # 每个文档的 id
        )

    def search(self, query, top_n):
        '''检索向量数据库'''
        results = self.collection.query(
            query_embeddings=self.embedding_fn([query]),
            n_results=top_n
        )
        return results


class RAG_Bot:
    def __init__(self, vector_db, llm_api, n_results=2):
        self.vector_db = vector_db
        self.llm_api = llm_api
        self.n_results = n_results

    def chat(self, user_query):
        # 1. 检索
        search_results = self.vector_db.search(user_query, self.n_results)
        # print(">>>>>>>>>documents【", search_results['documents'])

        # 2. 构建 Prompt
        prompt = build_prompt(
            prompt_template, context=search_results['documents'][0], query=user_query)

        # print(">>>>>", prompt)

        # 3. 调用 LLM
        response = self.llm_api(prompt)
        return response


def split_text(paragraphs, chunk_size=300, overlap_size=100):
    '''按指定 chunk_size 和 overlap_size 交叠割文本'''
    sentences = [s.strip() for p in paragraphs for s in sent_tokenize(p)]
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i]
        overlap = ''
        prev_len = 0
        prev = i - 1
        # 向前计算重叠部分
        while prev >= 0 and len(sentences[prev]) + len(overlap) <= overlap_size:
            overlap = sentences[prev] + ' ' + overlap
            prev -= 1
        chunk = overlap + chunk
        next = i + 1
        # 向后计算当前chunk
        while next < len(sentences) and len(sentences[next]) + len(chunk) <= chunk_size:
            chunk = chunk + ' ' + sentences[next]
            next += 1
        chunks.append(chunk)
        i = next
    return chunks


# 创建一个向量数据库对象
vector_db = MyVectorDBConnector(collection_name="demo_text_split",
                                embedding_fn=get_embeddings)

paragraphs = extract_text_from_pdf("llama2.pdf", page_numbers=[2, 3], min_line_length=10)
chunks = split_text(paragraphs, 300, 100)

# for p in paragraphs:
#     print(p + "\n")

# 向量数据库中创建文档
vector_db.add_documents(chunks)

# 创建一个RAG机器
rag_bot = RAG_Bot(vector_db, llm_api=get_completion)

# user_query = "llama 2 模型参数都有哪些？"

# search_results = vector_db.search(user_query, 2)
# for doc in search_results['documents'][0]:
#     print(doc + "\n")
#
# response = rag_bot.chat(user_query)
# print("====回复====")
# print(response)

if __name__ == '__main__':

    user_query = "llama 2安全吗？"
    search_results = vector_db.search(user_query, 5)

    for doc in search_results['documents'][0]:
        print(doc + "\n")

    response = rag_bot.chat(user_query)
    print("====回复====")
    print(response)

    #
    # user_query = "llama2有多少参数？"
    # results = vector_db.search(query=user_query, top_n=3)
    # #
    # for para in results["documents"][0]:
    #     print(para + "\n")

    # prompt = build_prompt(prompt_template, context=paragraphs, query="llama2是什么？")
    # print(get_completion(prompt))
    # 测试向量模型
    # test_query = ["测试文本"]
    # vec = get_embeddings(test_query)[0]
    # print(f"Total dimension: {len(vec)}")
    # print(f"First 10 elements: {vec[:10]}")

    # query = "国际争端"

    # documents = [
    #     "联合国就苏丹达尔富尔地区大规模暴力事件发出警告",
    #     "土耳其、芬兰、瑞典与北约代表将继续就瑞典“入约”问题进行谈判",
    #     "日本岐阜市陆上自卫队射击场内发生枪击事件 3人受伤",
    #     "国家游泳中心（水立方）：恢复游泳、嬉水乐园等水上项目运营",
    #     "我国首次在空间站开展舱外辐射生物学暴露实验",
    # ]
    #
    # query_vec = get_embeddings([query])[0]
    # doc_vecs = get_embeddings(documents)
    #
    # print("Query与自己的余弦距离: {:.2f}".format(cos_sim(query_vec, query_vec)))
    # print("Query与Documents的余弦距离:")
    # for vec in doc_vecs:
    #     print(cos_sim(query_vec, vec))
    #
    # print()
    #
    # print("Query与自己的欧氏距离: {:.2f}".format(l2(query_vec, query_vec)))
    # print("Query与Documents的欧氏距离:")
    # for vec in doc_vecs:
    #     print(l2(query_vec, vec))
