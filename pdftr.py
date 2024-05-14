import gradio as gr
import pdfplumber
from zhipuai import ZhipuAI


# def translate_pdf(file):
#     with pdfplumber.open(file) as pdf:
#         pages = pdf.pages
#         p1_text = pages[2].extract_text()
#
#     print(p1_text)
#
#     client = ZhipuAI(api_key="api_key")
#
#     response = client.chat.completions.create(
#         model="glm-4",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "你是一名翻译官。 将英文翻译成中文"
#             },
#             {
#                 "role": "user",
#                 "content": str(p1_text)
#
#             }
#         ],
#     )
#
#     content = response.choices[0].message.content
#     print(content)
#
#     return content
#     # 这里添加PDF翻译的逻辑
#
#     # return "Translated text"


import gradio as gr
import pdfplumber
from zhipuai import ZhipuAI


def translate_pdf(pdf_file, tranfer_style, source_lang, target_lang):
    with pdfplumber.open(pdf_file) as pdf:
        pages = pdf.pages
        p1_text = pages[2].extract_text()

    client = ZhipuAI(api_key="7aff1e84cbcb40fae39e9c9b26b0eaf1.xxxxxxx")

    response = client.chat.completions.create(
        model="glm-4",
        messages=[
            {
                "role": "system",
                "content": f"你是一名翻译风格为{tranfer_style}的翻译官。 将{source_lang}翻译成{target_lang}。"
            },
            {
                "role": "user",
                "content": str(p1_text)

            }
        ],
    )

    content = response.choices[0].message.content
    return content

def launch_gradio():
    iface = gr.Interface(
        fn=translate_pdf,
        title="OpenAI-Translator v2.0(PDF 电子书翻译工具)",
        inputs=[
            gr.File(label="上传PDF文件"),
            gr.Textbox(label="翻译风格（默认：小说）", placeholder="小说", value="小说"),
            gr.Textbox(label="源语言（默认：英文）", placeholder="English", value="English"),
            gr.Textbox(label="目标语言（默认：中文）", placeholder="Chinese", value="Chinese")
        ],
        outputs=[
            # gr.File(label="下载翻译文件")
            gr.Textbox(label="翻译结果")
        ],
        allow_flagging="never"
    )

    iface.launch(share=True)


if __name__ == "__main__":
    # file = "The_Old_Man_of_the_Sea.pdf"
    launch_gradio()
    # translate_pdf(file)