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
import gradio as gr
from sentence_transformers import CrossEncoder


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


def get_completion(prompt, model="glm-4-flash"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        stream=True,
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
    '''
    封装 zhipuAI 的 Embedding 模型接口，用于获取文本的向量表示。

    参数:
    - texts (list[str]): 需要转换为嵌入向量的文本列表。
    - model (str): 使用的嵌入模型，默认为 "embedding-2"。
    - dimensions (int, optional): 返回的嵌入向量维度，如果指定则按此维度返回，否则使用模型默认维度。

    返回:
    - list[list[float]]: 文本列表对应的嵌入向量列表。
    '''
    # 如果选择了模型"embedding-2"，则忽略用户自定义的dimensions参数
    if model == "embedding-2":
        dimensions = None

    # 根据是否指定dimensions调用不同的API方法
    if dimensions:
        # 使用指定维度的模型进行嵌入
        data = client.embeddings.create(input=texts, model=model, dimensions=dimensions).data
    else:
        # 使用默认维度的模型进行嵌入
        data = client.embeddings.create(input=texts, model=model).data

    # 提取每个数据项的embedding属性，形成最终的向量列表并返回
    return [x.embedding for x in data]


class MyVectorDBConnector:
    def __init__(self, collection_name, embedding_fn):
        '''
        初始化 VectorDB 连接器，创建或重置指定的 ChromaDB 集合，并设置嵌入函数。

        参数:
        - collection_name (str): ChromaDB 中集合的名称。
        - embedding_fn (callable): 用于生成文档嵌入向量的函数。
        '''
        # 初始化 ChromaDB 客户端，并允许重置数据库状态
        chroma_client = chromadb.Client(Settings(allow_reset=True))

        # 重置 ChromaDB 客户端状态，这将删除所有现有数据
        # chroma_client.reset()

        # 获取或创建指定名称的集合
        self.collection = chroma_client.get_or_create_collection(name=collection_name)

        # 保存用于生成嵌入向量的函数引用
        self.embedding_fn = embedding_fn

    def add_documents(self, documents):
        '''
        向当前集合中添加文档及其对应的嵌入向量。

        参数:
        - documents (list[str]): 待添加的文档列表。
        '''
        # 调用嵌入函数计算每个文档的向量
        # 添加文档、其向量及自动生成的id到集合中
        self.collection.add(
            embeddings=self.embedding_fn(documents),
            documents=documents,
            ids=[f"id{i}" for i in range(len(documents))]
        )

    def search(self, query, top_n):
        '''
        在集合中搜索与查询最相似的 top_n 个文档。

        参数:
        - query (str): 查询文本。
        - top_n (int): 需要返回的结果数量。

        返回:
        - dict: 包含查询结果的字典，包括与查询最相关的文档id、分数和文档内容。
        '''
        # 对查询文本应用嵌入函数以获取其向量表示
        # 执行查询并返回最相关的 top_n 个结果
        results = self.collection.query(
            query_embeddings=self.embedding_fn([query]),
            n_results=top_n
        )
        return results


class RAG_Bot:
    """
    RAG (Retrieval-Augmented Generation) 机器人类。

    该类使用检索增强的生成模型来进行聊天。

    参数:
    - vector_db: 向量数据库对象，用于检索相关文档。
    - llm_api: 大语言模型接口，用于生成回答。
    - n_results: 检索结果的数量，默认为2。
    """

    def __init__(self, vector_db, llm_api, n_results=2):
        self.vector_db = vector_db
        self.llm_api = llm_api
        self.n_results = n_results

    def chat(self, user_query):
        """
        与用户进行聊天。

        接收用户的查询，使用向量数据库进行检索，然后基于检索结果和用户查询构建prompt，
        最后通过大语言模型生成回答。

        参数:
        - user_query: 用户的查询字符串。

        返回:
        - response: 大语言模型生成的回答。
        """
        # 使用向量数据库检索与用户查询相关的文档
        # 1. 检索
        search_results = self.vector_db.search(user_query, self.n_results)

        # 根据检索结果和用户查询构建prompt
        # 2. 构建 Prompt
        prompt = build_prompt(
            prompt_template, context=search_results['documents'][0], query=user_query)

        # 使用大语言模型对构建的prompt进行处理，生成回答
        # 3. 调用 LLM
        response = self.llm_api(prompt)
        return response


def split_text(paragraphs, chunk_size=300, overlap_size=100):
    '''
    将文本按段落分割成句子，然后将句子重新组合成大小约为 `chunk_size` 的文本块，
    并确保相邻块之间有 `overlap_size` 的重叠内容。

    参数:
    - paragraphs (list[str]): 待分割的文本段落组成的列表。
    - chunk_size (int): 生成文本块的目标大小，默认为300字符。
    - overlap_size (int): 文本块之间的重叠字符数，默认为100字符。

    返回:
    - list[str]: 按照指定规则分割得到的文本块列表。
    '''
    # 使用sent_tokenize将段落分割成句子，并去除前后空白
    sentences = [s.strip() for p in paragraphs for s in sent_tokenize(p)]

    chunks = []  # 用于存储分割后的文本块
    i = 0  # 当前处理的句子索引

    while i < len(sentences):
        chunk = sentences[i]  # 当前块起始于当前句子
        overlap = ''  # 初始化重叠部分为空

        # 向前构建重叠部分，直到达到overlap_size或到头
        prev = i - 1
        while prev >= 0 and len(sentences[prev]) + len(overlap) <= overlap_size:
            overlap = sentences[prev] + ' ' + overlap
            prev -= 1

        # 将重叠部分添加到当前块的开始
        chunk = overlap + chunk

        next = i + 1  # 准备向后扩展块

        # 向后扩展当前块，直到达到chunk_size或到尾
        while next < len(sentences) and len(sentences[next]) + len(chunk) <= chunk_size:
            chunk = chunk + ' ' + sentences[next]
            next += 1

        # 将构建好的块添加到结果列表
        chunks.append(chunk)

        # 更新索引到下一个未处理的句子
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

def bot_response(user_query, chat_history):
    """
    接收用户输入，记录对话历史，并返回机器人的响应。

    参数:
    - user_query: 用户的当前查询字符串。
    - chat_history: 对话的历史记录列表，每项为元组(用户输入, 机器人响应)。

    返回:
    - 更新后的对话历史，包含最新的用户输入和机器人的响应。
    """
    response = rag_bot.chat(user_query)
    chat_history.append((user_query, response))
    return "", chat_history  # 返回空字符串作为新输入的提示，以及更新的聊天历史


if __name__ == '__main__':
    # 混合检索
    # query = "how safe is llama 2?"
    # documents = [
    #     "玛丽患有肺癌，癌细胞已转移",
    #     "刘某肺癌I期",
    #     "张某经诊断为非小细胞肺癌III期",
    #     "小细胞肺癌是肺癌的一种"
    # ]
    #
    # query_vec = get_embeddings([query])[0]
    # doc_vecs = get_embeddings(documents)
    #
    # print("Cosine distance:")
    # for vec in doc_vecs:
    #     print(cos_sim(query_vec, vec))

    # --------------------------------------------------------------------------------------
    # user_query = "how safe is llama 2?"
    # # user_query = "llama 2都有哪些参数？"
    # search_results = vector_db.search(user_query, 5)
    #
    # model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)  # 英文，模型较小
    #
    # scores = model.predict([(doc, user_query) for doc in search_results['documents'][0]])
    #
    # # 按得分排序
    # sorted_list = sorted(
    #     zip(scores, search_results['documents'][0]), key=lambda x: x[0], reverse=True)
    #
    # for score, doc in sorted_list:
    #     print(f"{score}\t{doc}\n")

    # ------------------------------------------------------------------------------------------
    user_query = "llama 2都有哪些参数？"
    search_results = vector_db.search(user_query, 5)
    for doc in search_results['documents'][0]:
        print(doc + "\n")

    response = rag_bot.chat(user_query)
    print("====回复====")
    print(response)
    # --------------------------------------------------------------------------------------------
    # with gr.Blocks() as demo:
    #     chatbot = gr.Chatbot()  # 创建聊天机器人组件
    #     message = gr.Textbox(label="用户输入")  # 用户输入框
    #
    #     # 设置Gradio界面的交互逻辑
    #     message.submit(bot_response, [message, chatbot], [message, chatbot])
    #
    # demo.launch()
    # --------------------------------------------------------------------------------------------

    #
    # user_query = "llama2有多少参数？"
    # results = vector_db.search(query=user_query, top_n=3)
    # #
    # for para in results["documents"][0]:
    #     print(para + "\n")
    #
    # prompt = build_prompt(prompt_template, context=paragraphs, query="llama2是什么？")
    # print(get_completion(prompt))
    # --------------------------------------------------------------------------------------------
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
