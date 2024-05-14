# pip install zhipuai 请先在终端进行安装

from zhipuai import ZhipuAI

client = ZhipuAI(api_key="7aff1e84cbcb40fae39e9c9b26b0eaf1.xxxxxxx")

response = client.chat.completions.create(
    model="glm-4",
    messages=[
        {
            "role": "system",
            "content": "你是一名翻译官。 将英文翻译成中文"
        },
        {
            "role": "user",
            "content": "你好"

        }
    ],
)


if __name__ == '__main__':
    print(response.choices[0].message)