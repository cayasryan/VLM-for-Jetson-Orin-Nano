import gradio as gr
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "SmolVLM-Base" 

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE)

def add_prompt(prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]
        },
    ]

    return messages

# Gradio interface
with gr.Blocks() as demo:
    gr.HTML(
        """
        <h1 style='text-align: center'>
        Visual Language Model for Jetson Orin Nano
        </h1>
        <p style='text-align: center'>
        Capture a snapshot by clicking on the camera icon in the Webcam Feed then ask the VLM about the image!
        </p>
        """
    )
    
    with gr.Row():
        # Webcam streaming
        input_img = gr.Image(sources=["webcam"], type="pil", label="Webcam Feed")
    with gr.Column():
        with gr.Row():
            # Prompt input
            prompt_input = gr.Textbox(label="Enter Prompt")
            output_text = gr.Textbox(label="Model Output", interactive=False)

    with gr.Row():
        ask_prompt_btn = gr.Button("Ask Prompt")
        gr.ClearButton([input_img, prompt_input, output_text])

    # Define interaction
    def process_inputs(image, prompt):
        if image is None:
            return "No image captured. Please capture a snapshot first."
        
        messages = add_prompt(prompt)
        final_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=final_prompt, images=[image], return_tensors="pt").to(DEVICE)

        generated_ids = model.generate(**inputs, max_new_tokens=10)
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return generated_texts[0]

    # When the Capture Snapshot button is clicked, set the value to True
    ask_prompt_btn.click(process_inputs, inputs=[input_img, prompt_input], outputs=[output_text])

demo.launch()
