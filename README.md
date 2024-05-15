pdftr.py：
1.支持图形用户界面（GUI），提升易用性。添加对保留源 PDF 的原始布局的支持。添加对其他语言的支持（需要手动输入）。
2.在 openai-translator gradio 图形化界面基础上，支持风格化翻译，如：小说、新闻稿、作家风格等（需要手动输入）。
3.基于 ChatGLM2-6B 实现带图形化界面的 openai-translator

character_glm_demo.py：
实现 role-play 对话数据生成工具，要求包含下列功能：
基于一段文本（自己找一段文本，复制到提示词就可以了，比如你可以从小说中选取一部分文本，注意文本要用 markdown 格式）生成角色人设，可借助 ChatGLM 实现。
给定两个角色的人设，调用 CharacterGLM 交替生成他们的回复。
将生成的对话数据保存到文件中。
