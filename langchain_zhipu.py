import os
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import List, Dict, Optional
from enum import Enum
import json
from config import API_KEY


# 输出结构
class SortEnum(str, Enum):
    data = 'data'
    price = 'price'


class OrderingEnum(str, Enum):
    ascend = 'ascend'
    descend = 'descend'


class Semantics(BaseModel):
    name: Optional[str] = Field(description="流量包名称", default=None)
    price_lower: Optional[int] = Field(description="价格下限", default=None)
    price_upper: Optional[int] = Field(description="价格上限", default=None)
    data_lower: Optional[int] = Field(description="流量下限", default=None)
    data_upper: Optional[int] = Field(description="流量上限", default=None)
    sort_by: Optional[SortEnum] = Field(description="按价格或流量排序", default=None)
    ordering: Optional[OrderingEnum] = Field(description="升序或降序排列", default=None)


# OutputParser
parser = PydanticOutputParser(pydantic_object=Semantics)

# Prompt 模板
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "将用户的输入解析成JSON表示。输出格式如下：\n{format_instructions}\n不要输出未提及的字段。",
        ),
        ("human", "{text}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# 模型
# model = ChatOpenAI(model="gpt-4-turbo", temperature=0)
model = ChatOpenAI(
    temperature=0.95,
    model="glm-4",
    openai_api_key=API_KEY,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)
# LCEL 表达式
runnable = (
    {"text": RunnablePassthrough()} | prompt | model | StrOutputParser()
)

# 运行
for s in runnable.stream("不超过100元的流量大的套餐有哪些"):
    print(s, end="")



