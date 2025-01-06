# Gradio demo of StyleTTS 2 by @fakerybakery
import gradio as gr
import msinference
import ljinference
import torch
import os
from tortoise.utils.text import split_and_recombine_text
import numpy as np
import pickle
theme = gr.themes.Base(
    font=[gr.themes.GoogleFont('Libre Franklin'), gr.themes.GoogleFont('Public Sans'), 'system-ui', 'sans-serif'],
)
voicelist = ['f-us-1', 'f-us-2', 'f-us-3', 'f-us-4', 'm-us-1', 'm-us-2', 'm-us-3', 'm-us-4']
voices = {}
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)
# todo: cache computed style, load using pickle
# if os.path.exists('voices.pkl'):
    # with open('voices.pkl', 'rb') as f:
        # voices = pickle.load(f)
# else:
for v in voicelist:
    voices[v] = msinference.compute_style(f'voices/{v}.wav')
def synthesize(text, voice, lngsteps, password, progress=gr.Progress()):
    if text.strip() == "":
        raise gr.Error("You must enter some text")
    if lngsteps > 25:
        raise gr.Error("Max 25 steps")
    if lngsteps < 5:
        raise gr.Error("Min 5 steps")
    texts = split_and_recombine_text(text)
    v = voice.lower()
    audios = []
    for t in progress.tqdm(texts):
        audios.append(msinference.inference(t, voices[v], alpha=0.3, beta=0.7, diffusion_steps=lngsteps, embedding_scale=1))
    return (24000, np.concatenate(audios))
def clsynthesize(text, voice, vcsteps):
    if text.strip() == "":
        raise gr.Error("You must enter some text")
    # if global_phonemizer.phonemize([text]) > 300:
    if len(text) > 400:
        raise gr.Error("Text must be under 400 characters")
    return (24000, msinference.inference(text, msinference.compute_style(voice), alpha=0.3, beta=0.7, diffusion_steps=vcsteps, embedding_scale=1))
def ljsynthesize(text):
    if text.strip() == "":
        raise gr.Error("You must enter some text")
    # if global_phonemizer.phonemize([text]) > 300:
    if len(text) > 400:
        raise gr.Error("Text must be under 400 characters")
    noise = torch.randn(1,1,256).to('cuda' if torch.cuda.is_available() else 'cpu')
    return (24000, ljinference.inference(text, noise, diffusion_steps=7, embedding_scale=1))


with gr.Blocks() as vctk: # just realized it isn't vctk but libritts but i'm too lazy to change it rn
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(label="Văn bản", info="Bạn muốn đọc gì? Công cụ AI này hoạt động tốt hơn với các câu dài.", interactive=True)
            voice = gr.Dropdown(voicelist, label="Giọng nói", info="Chọn giọng nói tham khảo", value='m-us-2', interactive=True)
            multispeakersteps = gr.Slider(minimum=5, maximum=15, value=7, step=1, label="Số bước khuếch tán", info="Càng cao thì chất lượng càng tốt, nhưng chậm hơn", interactive=True)
        with gr.Column(scale=1):
            btn = gr.Button("Tạo giọng nói", variant="primary")
            audio = gr.Audio(interactive=False, label="Kết quả")
            btn.click(synthesize, inputs=[inp, voice, multispeakersteps], outputs=[audio], concurrency_limit=4)
with gr.Blocks() as clone:
    with gr.Row():
        with gr.Column(scale=1):
            clinp = gr.Textbox(label="Văn bản", info="Bạn muốn đọc gì? Công cụ AI này hoạt động tốt hơn với các câu hoàn chỉnh.", interactive=True)
            clvoice = gr.Audio(label="Giọng nói", interactive=True, type='filepath', max_length=300)
            vcsteps = gr.Slider(minimum=5, maximum=40, value=20, step=1, label="Số bước lặp AI", info="Càng cao thì chất lượng càng tốt, nhưng chậm hơn", interactive=True)
        with gr.Column(scale=1):
            clbtn = gr.Button("Tạo giọng nói", variant="primary")
            claudio = gr.Audio(interactive=False, label="Kết quả")
            clbtn.click(clsynthesize, inputs=[clinp, clvoice, vcsteps], outputs=[claudio], concurrency_limit=4)
with gr.Blocks() as lj:
    with gr.Row():
        with gr.Column(scale=1):
            ljinp = gr.Textbox(label="Văn bản", info="Bạn muốn đọc gì? StyleTTS2 hoạt động tốt hơn với các câu hoàn chỉnh.", interactive=True)
        with gr.Column(scale=1):
            ljbtn = gr.Button("Tạo giọng nói", variant="primary")
            ljaudio = gr.Audio(interactive=False, label="Kết quả")
            ljbtn.click(ljsynthesize, inputs=[ljinp], outputs=[ljaudio], concurrency_limit=4)
with gr.Blocks(title="VNUIS IDT - Công cụ AI sao chép giọng nói StyleTTS 2", css="footer{display:none !important}", theme=theme) as demo:
    gr.Markdown("""# VNUIS IDT - Công cụ AI sao chép giọng nói StyleTTS 2

Triển khai bởi Nhóm Nghiên cứu Đổi mới sáng tạo và Chuyển đổi số, Trường Quốc Tế, Đại học Quốc gia Hà Nội

**Cảnh báo quan trọng**

Công cụ AI này chỉ dành cho việc tổng hợp giọng nói cá nhân của bạn. Bạn hoàn toàn chịu trách nhiệm về việc sử dụng công cụ này.

**Lưu ý:**

1. Bạn không được phép sử dụng giọng nói của người khác để sao chép hoặc tạo giọng nói mà không có sự cho phép.
2. Bất kỳ hành vi vi phạm nào liên quan đến quyền riêng tư hoặc bản quyền sẽ do bạn chịu trách nhiệm trước pháp luật.

**Công cụ AI này hoạt động tốt hơn với các đoạn văn bản dài.** Ví dụ, yêu cầu nói "Hello" sẽ cho ra kết quả chất lượng thấp hơn so với việc yêu cầu một đoạn dài.""")
    gr.TabbedInterface([clone], ['Sao chép giọng nói'])
    gr.Markdown("""
[Paper](https://arxiv.org/abs/2306.07691) - [Samples](https://styletts2.github.io/) - [Code](https://github.com/yl4579/StyleTTS2) - [Interface](https://twitter.com/realmrfakename).

**Nhóm Nghiên cứu Đổi mới sáng tạo và Chuyển đổi số, Trường Quốc Tế, Đại học Quốc gia Hà Nội**
""")
if __name__ == "__main__":
    demo.queue(api_open=False, max_size=15).launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssl_certfile="ssl/ssl.cert",
        ssl_keyfile="ssl/ssl.key",
        ssl_verify=False,
        show_api=False,
        share=False
    )

