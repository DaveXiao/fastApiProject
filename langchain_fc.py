from langchain_core.tools import tool
from config import API_KEY
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

model = ChatOpenAI(
    temperature=0.95,
    model="glm-4",
    openai_api_key=API_KEY,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)


@tool
def multiply(first_int: int, second_int: int) -> int:
    """两个整数相乘"""
    return first_int * second_int


@tool
def add(first_int: int, second_int: int) -> int:
    """Add two integers."""
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    """Exponentiate the base to the exponent power."""
    return base ** exponent


from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import JsonOutputToolsParser

tools = [multiply, add, exponentiate]
# 带有分支的 LCEL
llm_with_tools = model.bind_tools(tools) | {
    "functions": JsonOutputToolsParser(),
    "text": StrOutputParser()
}

result = llm_with_tools.invoke("1024的16倍是多少")

print(result)

result = llm_with_tools.invoke("你是谁")
print(result)


from typing import Union
from operator import itemgetter
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)

# 名称到函数的映射
tool_map = {tool.name: tool for tool in tools}


def call_tool(tool_invocation: dict) -> Union[str, Runnable]:
    """根据模型选择的 tool 动态创建 LCEL"""
    tool = tool_map[tool_invocation["type"]]
    return RunnablePassthrough.assign(
        output=itemgetter("args") | tool
    )


# .map() 使 function 逐一作用与一组输入
call_tool_list = RunnableLambda(call_tool).map()

import json


def route(response):
    if len(response["functions"]) > 0:
        return response["functions"]
    else:
        return response["text"]


llm_with_tools = model.bind_tools(tools) | {
    "functions": JsonOutputToolsParser() | call_tool_list,
    "text": StrOutputParser()
} | RunnableLambda(route)

result = llm_with_tools.invoke("1024的16倍是多少")
print(result)

result = llm_with_tools.invoke("你好")
print(result)