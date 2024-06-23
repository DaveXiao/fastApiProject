import os
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from config import API_KEY

os.environ["TAVILY_API_KEY"] = API_KEY

llm = ChatOpenAI(
    temperature=0.95,
    model="glm-4",
    openai_api_key=API_KEY,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

tools = [TavilySearchResults(max_results=2)]
prompt = hub.pull("hwchase17/react")

# Choose the LLM to use
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "LangChain是什么?"})