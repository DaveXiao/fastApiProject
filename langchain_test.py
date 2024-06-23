# 导入chat model即将使用的 prompt templates
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chat_models import ChatOpenAI

# from langchain.chains import LLMChain
# from langchain.memory import ConversationBufferMemory
from config import API_KEY

# 翻译任务指令始终由 System 角色承担
template = (
    """You are a translation expert, proficient in various languages. \n
    Translates English to Chinese."""
)
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

print(system_message_prompt)

# 待翻译文本由 Human 角色输入
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)


# 使用 System 和 Human 角色的提示模板构造 ChatPromptTemplate
chat_prompt_template = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)


# 生成聊天模型真正可用的消息记录 Messages
chat_prompt = chat_prompt_template.format_prompt(text="I love programming.").to_messages()


# 为了翻译结果的稳定性，将 temperature 设置为 0
# translation_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
translation_model = ChatOpenAI(
    temperature=0.95,
    model="glm-4",
    openai_api_key=API_KEY,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

translation_result = translation_model(chat_prompt)

