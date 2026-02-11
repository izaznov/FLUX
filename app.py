import os
import subprocess
import sys
import io
import gradio as gr
import numpy as np
import random
import spaces
import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from diffusers import BitsAndBytesConfig as DiffBitsAndBytesConfig
import requests
from PIL import Image
import json
import base64
from huggingface_hub import InferenceClient

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

hf_client = InferenceClient(
    api_key=os.environ.get("HF_TOKEN"),
)
VLM_MODEL = "baidu/ERNIE-4.5-VL-424B-A47B-Base-PT"

SYSTEM_PROMPT_TEXT_ONLY = """You are an expert prompt engineer for FLUX.2 by Black Forest Labs. Rewrite user prompts to be more descriptive while strictly preserving their core subject and intent.

Guidelines:
1. Structure: Keep structured inputs structured (enhance within fields). Convert natural language to detailed paragraphs.
2. Details: Add concrete visual specifics - form, scale, textures, materials, lighting (quality, direction, color), shadows, spatial relationships, and environmental context.
3. Text in Images: Put ALL text in quotation marks, matching the prompt's language. Always provide explicit quoted text for objects that would contain text in reality (signs, labels, screens, etc.) - without it, the model generates gibberish.

Output only the revised prompt and nothing else."""

SYSTEM_PROMPT_WITH_IMAGES = """You are FLUX.2 by Black Forest Labs, an image-editing expert. You convert editing requests into one concise instruction (50-80 words, ~30 for brief requests).

Rules:
- Single instruction only, no commentary
- Use clear, analytical language (avoid "whimsical," "cascading," etc.)
- Specify what changes AND what stays the same (face, lighting, composition)
- Reference actual image elements
- Turn negatives into positives ("don't change X" → "keep X")
- Make abstractions concrete ("futuristic" → "glowing cyan neon, metallic panels")
- Keep content PG-13

Output only the final instruction in plain text and nothing else."""

def remote_text_encoder(prompts):
    from gradio_client import Client
    
    client = Client("multimodalart/mistral-text-encoder")
    result = client.predict(
        prompt=prompts,
        api_name="/encode_text"
    )
    
    # Load returns a tensor, usually on CPU by default
    prompt_embeds = torch.load(result[0])
    return prompt_embeds

# Load model
repo_id = "black-forest-labs/FLUX.2-dev"

dit = Flux2Transformer2DModel.from_pretrained(
    repo_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)

pipe = Flux2Pipeline.from_pretrained(
    repo_id,
    text_encoder=None,
    transformer=dit,
    torch_dtype=torch.bfloat16
)
pipe.to(device)

# Pull pre-compiled Flux2 Transformer blocks from HF hub
spaces.aoti_blocks_load(pipe.transformer, "zerogpu-aoti/FLUX.2", variant="fa3")

def image_to_data_uri(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def upsample_prompt_logic(prompt, image_list):
    try:
        if image_list and len(image_list) > 0:
            # Image + Text Editing Mode
            system_content = SYSTEM_PROMPT_WITH_IMAGES
            
            # Construct user message with text and images
            user_content = [{"type": "text", "text": prompt}]
            
            for img in image_list:
                data_uri = image_to_data_uri(img)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri}
                })
                
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
        else:
            # Text Only Mode
            system_content = SYSTEM_PROMPT_TEXT_ONLY
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]

        completion = hf_client.chat.completions.create(
            model=VLM_MODEL,
            messages=messages,
            max_tokens=1024
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Upsampling failed: {e}")
        return prompt

def update_dimensions_from_image(image_list):
    """Update width/height sliders based on uploaded image aspect ratio.
    Keeps one side at 1024 and scales the other proportionally, with both sides as multiples of 8."""
    if image_list is None or len(image_list) == 0:
        return 1024, 1024  # Default dimensions
    
    # Get the first image to determine dimensions
    img = image_list[0][0]  # Gallery returns list of tuples (image, caption)
    img_width, img_height = img.size
    
    aspect_ratio = img_width / img_height
    
    if aspect_ratio >= 1:  # Landscape or square
        new_width = 1024
        new_height = int(1024 / aspect_ratio)
    else:  # Portrait
        new_height = 1024
        new_width = int(1024 * aspect_ratio)
    
    # Round to nearest multiple of 8
    new_width = round(new_width / 8) * 8
    new_height = round(new_height / 8) * 8
    
    # Ensure within valid range (minimum 256, maximum 1024)
    new_width = max(256, min(1024, new_width))
    new_height = max(256, min(1024, new_height))
    
    return new_width, new_height

# Updated duration function to match generate_image arguments (including progress)
def get_duration(prompt_embeds, image_list, width, height, num_inference_steps, guidance_scale, seed, progress=gr.Progress(track_tqdm=True)):
    num_images = 0 if image_list is None else len(image_list)
    step_duration = 1 + 0.8 * num_images
    return max(65, num_inference_steps * step_duration + 10)

@spaces.GPU(duration=get_duration)
def generate_image(prompt_embeds, image_list, width, height, num_inference_steps, guidance_scale, seed, progress=gr.Progress(track_tqdm=True)):
    # Move embeddings to GPU only when inside the GPU decorated function
    prompt_embeds = prompt_embeds.to(device)
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    pipe_kwargs = {
        "prompt_embeds": prompt_embeds,
        "image": image_list,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "width": width,
        "height": height,
    }
    
    # Progress bar for the actual generation steps
    if progress:
        progress(0, desc="Starting generation...")
        
    image = pipe(**pipe_kwargs).images[0]
    return image

def infer(prompt, input_images=None, seed=42, randomize_seed=False, width=1024, height=1024, num_inference_steps=50, guidance_scale=2.5, prompt_upsampling=False, progress=gr.Progress(track_tqdm=True)):
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    # Prepare image list (convert None or empty gallery to None)
    image_list = None
    if input_images is not None and len(input_images) > 0:
        image_list = []
        for item in input_images:
            image_list.append(item[0])

    # 1. Upsampling (Network bound - No GPU needed)
    final_prompt = prompt
    if prompt_upsampling:
        progress(0.05, desc="Upsampling prompt...")
        final_prompt = upsample_prompt_logic(prompt, image_list)
        print(f"Original Prompt: {prompt}")
        print(f"Upsampled Prompt: {final_prompt}")

    # 2. Text Encoding (Network bound - No GPU needed)
    progress(0.1, desc="Encoding prompt...")
    # This returns CPU tensors
    prompt_embeds = remote_text_encoder(final_prompt)
    
    # 3. Image Generation (GPU bound)
    progress(0.3, desc="Waiting for GPU...")
    image = generate_image(
        prompt_embeds, 
        image_list, 
        width, 
        height, 
        num_inference_steps, 
        guidance_scale, 
        seed, 
        progress
    )
    
    return image, seed

examples = [
    ["Create a vase on a table in living room, the color of the vase is a gradient of color, starting with #02eb3c color and finishing with #edfa3c. The flowers inside the vase have the color #ff0088"],
    ["Photorealistic infographic showing the complete Berlin TV Tower (Fernsehturm) from ground base to antenna tip, full vertical view with entire structure visible including concrete shaft, metallic sphere, and antenna spire. Slight upward perspective angle looking up toward the iconic sphere, perfectly centered on clean white background. Left side labels with thin horizontal connector lines: the text '368m' in extra large bold dark grey numerals (#2D3748) positioned at exactly the antenna tip with 'TOTAL HEIGHT' in small caps below. The text '207m' in extra large bold with 'TELECAFÉ' in small caps below, with connector line touching the sphere precisely at the window level. Right side label with horizontal connector line touching the sphere's equator: the text '32m' in extra large bold dark grey numerals with 'SPHERE DIAMETER' in small caps below. Bottom section arranged in three balanced columns: Left - Large text '986' in extra bold dark grey with 'STEPS' in caps below. Center - 'BERLIN TV TOWER' in bold caps with 'FERNSEHTURM' in lighter weight below. Right - 'INAUGURATED' in bold caps with 'OCTOBER 3, 1969' below. All typography in modern sans-serif font (such as Inter or Helvetica), color #2D3748, clean minimal technical diagram style. Horizontal connector lines are thin, precise, and clearly visible, touching the tower structure at exact corresponding measurement points. Professional architectural elevation drawing aesthetic with dynamic low angle perspective creating sense of height and grandeur, poster-ready infographic design with perfect visual hierarchy."],
    ["Soaking wet capybara taking shelter under a banana leaf in the rainy jungle, close up photo"],
    ["A kawaii die-cut sticker of a chubby orange cat, featuring big sparkly eyes and a happy smile with paws raised in greeting and a heart-shaped pink nose. The design should have smooth rounded lines with black outlines and soft gradient shading with pink cheeks."],
]

examples_images = [
    # ["Replace the top of the person from image 1 with the one from image 2", ["person1.webp", "woman2.webp"]],
    ["The person from image 1 is petting the cat from image 2, the bird from image 3 is next to them", ["woman1.webp", "cat_window.webp", "bird.webp"]]
]

css="""
#col-container {
    margin: 0 auto;
    max-width: 1200px;
}
.gallery-container img{
    object-fit: contain;
}
"""

with gr.Blocks() as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# FLUX.2 [dev]
FLUX.2 [dev] is a 32B model rectified flow capable of generating, editing and combining images based on text instructions model [[model](https://huggingface.co/black-forest-labs/FLUX.2-dev)], [[blog](https://bfl.ai/blog/flux-2)]
        """)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    prompt = gr.Text(
                        label="Prompt",
                        show_label=False,
                        max_lines=2,
                        placeholder="Enter your prompt",
                        container=False,
                        scale=3
                    )
                    
                    run_button = gr.Button("Run", scale=1)
                    
                with gr.Accordion("Input image(s) (optional)", open=True):
                    input_images = gr.Gallery(
                        label="Input Image(s)",
                        type="pil",
                        columns=3,
                        rows=1,
                    )
                
                with gr.Accordion("Advanced Settings", open=False):
                    prompt_upsampling = gr.Checkbox(
                        label="Prompt Upsampling",
                        value=True,
                        info="Automatically enhance the prompt using a VLM"
                    )
        
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0,
                    )
                    
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    
                    with gr.Row():
                        
                        width = gr.Slider(
                            label="Width",
                            minimum=256,
                            maximum=MAX_IMAGE_SIZE,
                            step=8,
                            value=1024,
                        )
                        
                        height = gr.Slider(
                            label="Height",
                            minimum=256,
                            maximum=MAX_IMAGE_SIZE,
                            step=8,
                            value=1024,
                        )
                    
                    with gr.Row():
                        
                        num_inference_steps = gr.Slider(
                            label="Number of inference steps",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=30,
                        )
                        
                        guidance_scale = gr.Slider(
                            label="Guidance scale",
                            minimum=0.0,
                            maximum=10.0,
                            step=0.1,
                            value=4,
                        )
                
                
            with gr.Column():
                result = gr.Image(label="Result", show_label=False)
            
        
        gr.Examples(
            examples=examples,
            fn=infer,
            inputs=[prompt],
            outputs=[result, seed],
            cache_examples=True,
            cache_mode="lazy"
        )

        gr.Examples(
            examples=examples_images,
            fn=infer,
            inputs=[prompt, input_images],
            outputs=[result, seed],
            cache_examples=True,
            cache_mode="lazy"
        )

    # Auto-update dimensions when images are uploaded
    input_images.upload(
        fn=update_dimensions_from_image,
        inputs=[input_images],
        outputs=[width, height]
    )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[prompt, input_images, seed, randomize_seed, width, height, num_inference_steps, guidance_scale, prompt_upsampling],
        outputs=[result, seed]
    )

demo.launch(css=css)