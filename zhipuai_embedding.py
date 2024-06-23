from zhipuai import ZhipuAI
from config import API_KEY

client = ZhipuAI(api_key=API_KEY)


response = client.embeddings.create(
    model="embedding-2",  # 填写需要调用的模型名称
    input="你好",
)

print(response.data)
