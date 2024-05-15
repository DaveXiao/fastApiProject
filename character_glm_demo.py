import os
from api import generate_role_appearance, get_characterglm_response
from data_types import TextMsg, TextMsgList, CharacterMeta


# 文章段落
text_description = """
玄德曰：“汉室倾颓，奸臣窃命，主上蒙尘。孤不度德量力，欲信大义于天下，而智术短浅，遂用猖蹶，至于今日。然志犹未已，君谓计将安出？”
孔明曰：“自董卓造逆以来，天下豪杰并起。曹操势不及袁绍，而竟能克绍者，非惟天时，抑亦人谋也。今操已拥百万之众，挟天子而令诸侯，此诚不可与争锋。孙权据有江东，已历三世，国险而民附，贤能为之用，此可与为援而不可图也。荆州北据汉沔，利尽南海，东连吴会，西通巴蜀，此用武之国，而其主不能守，此殆天所以资将军也。益州险塞，沃野千里，天府之土，高祖因之以成帝业。刘璋暗弱，张鲁在北，民殷国富而不知存恤，智能之士思得明君。将军既帝室之胄，信义著于四海，总揽英雄，思贤如渴，若跨有荆、益，保其岩阻，西和诸戎，南抚夷越，外结孙权，内修政理；天下有变，则命一上将将荆州之军以向宛、洛，将军身率益州之众出于秦川，百姓孰敢不箪食壶浆以迎将军者乎？诚如是，则霸业可成，汉室可兴矣。”
"""

# 使用generate_role_appearance生成角色人设
role_appearance_gen = generate_role_appearance(text_description)

# 假设我们从生成器中获取了第一个结果作为刘备的角色人设
liu_bei_appearance = next(role_appearance_gen)

# 假设我们从生成器中获取了第二个结果作为诸葛亮的角色人设
zhuge_liang_appearance = next(role_appearance_gen)


# 假设我们从生成器中获取了第一个结果作为人设描述
bot_appearance = liu_bei_appearance
user_appearance = zhuge_liang_appearance

# 创建CharacterMeta对象
bot_meta = CharacterMeta(bot_name="诸葛亮", bot_info=bot_appearance, user_name="刘备", user_info=user_appearance)
user_meta = CharacterMeta(bot_name="刘备", bot_info=user_appearance, user_name="诸葛亮", user_info=bot_appearance)

# 初始化对话消息列表
messages = []

# 添加第一个消息作为起始点
messages.append(TextMsg(role="user", content="隆中对，三国时期的重要战略对话。请根据给定的上下文，生成回复。"))

# 模拟对话，交替生成回复
for i in range(10):  # 假设生成5轮对话
    if i % 2 == 0:
        current_meta = user_meta
        last_msg = messages[-1]  # 获取上一条消息作为当前角色的回复
    else:
        current_meta = bot_meta

    # 使用CharacterGLM生成回复
    response = get_characterglm_response([TextMsg(role="user", content=last_msg['content'])], current_meta)
    cleaned_text = ''
    # 将生成的回复添加到对话历史中
    for reply in response:
        cleaned_text += ''.join(reply)
    reply_msg = TextMsg(role="assistant", content=cleaned_text, meta=current_meta)
    messages.append(reply_msg)


# 将对话保存到文件
def save_to_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for msg in data:
            if msg['role'] == 'user':
                file.write(f"{current_meta['bot_name']}: {msg['content']}\n")
            elif msg['role'] == 'assistant':
                file.write(f"{msg['meta']['bot_name']}: {msg['content']}\n")

# 保存对话到文件
save_to_file(messages[1:], "dialogue_history_01.txt")

print("对话已生成并保存到文件中。")
