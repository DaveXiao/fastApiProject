from langchain import hub
import json

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from config import API_KEY

from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import Tool, tool

import calendar
import dateutil.parser as parser
from datetime import date

# 自定义工具


search = SerpAPIWrapper()
tools = [
    Tool.from_function(
        func=search.run,
        name="Search",
        description="useful for when you need to answer questions about current events"
    ),
]


@tool("weekday")
def weekday(date_str: str) -> str:
    """Convert date to weekday name"""
    d = parser.parse(date_str)
    return calendar.day_name[d.weekday()]


tools += [weekday]

# 下载一个现有的 Prompt 模板
prompt = hub.pull("hwchase17/react")

print(prompt.template)

llm = ChatOpenAI(
    temperature=0.95,
    model="glm-4",
    openai_api_key=API_KEY,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

# 定义一个 agent: 需要大模型、工具集、和 Prompt 模板
agent = create_react_agent(llm, tools, prompt)
# 定义一个执行器：需要 agent 对象 和 工具集
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 执行
agent_executor.invoke({"input": "周杰伦出生那天是星期几"})
