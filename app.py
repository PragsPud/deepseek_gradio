from threading import Thread
import torch
import spaces
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

MODEL_NAME = MODEL_ID.split('/')[1]
CHAT_TEMPLATE = "َAuto"
CONTEXT_LENGTH = 16000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto"
)

@spaces.GPU()
def predict(message, history, system_prompt, temperature, max_new_tokens, top_k, repetition_penalty, top_p):
    stop_tokens = ["<|endoftext|>", "<|im_end|>","|im_end|"]
    instruction = '<|im_start|>system\n' + system_prompt + '\n<|im_end|>\n'
    for user, assistant in history:
        instruction += f'<|im_start|>user\n{user}\n<|im_end|>\n<|im_start|>assistant\n{assistant}\n<|im_end|>\n'
    instruction += f'<|im_start|>user\n{message}\n<|im_end|>\n<|im_start|>assistant\n'
    
    print(instruction)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    enc = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True)
    input_ids, attention_mask = enc.input_ids, enc.attention_mask

    if input_ids.shape[1] > CONTEXT_LENGTH:
        input_ids = input_ids[:, -CONTEXT_LENGTH:]
        attention_mask = attention_mask[:, -CONTEXT_LENGTH:]

    generate_kwargs = dict(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        streamer=streamer,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        top_p=top_p
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    outputs = []
    for new_token in streamer:
        outputs.append(new_token)
        if new_token in stop_tokens:
            
            break
        yield "".join(outputs)

gr.ChatInterface(
    predict,
    title=MODEL_NAME + " DEMO",
     
    additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False),
    additional_inputs=[
        gr.Textbox("You are a useful assistant. First, recognize the user's request. Then, think and reply carefully.", label="System prompt"),
        gr.Slider(0, 1, 0.1, label="Temperature"),
        gr.Slider(0, 30000, 200, label="Max new tokens"),
        gr.Slider(1, 80, 40, label="Top K sampling"),
        gr.Slider(0, 2, 1.1, label="Repetition penalty"),
        gr.Slider(0, 1, 0.95, label="Top P sampling"),
    ],
).queue().launch()