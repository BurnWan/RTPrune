from transformers import AutoModel, AutoTokenizer
import torch
import os
import re
from PIL import Image
from pathlib import Path
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def clean_formula(text):

    formula_pattern = r'\\\[(.*?)\\\]'
    
    def process_formula(match):
        formula = match.group(1)

        formula = re.sub(r'\\quad\s*\([^)]*\)', '', formula)
        
        formula = formula.strip()
        
        return r'\[' + formula + r'\]'

    cleaned_text = re.sub(formula_pattern, process_formula, text)
    
    return cleaned_text

def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)


    # mathes_image = []
    mathes_other = []
    for a_match in matches:
        mathes_other.append(a_match[0])
    return matches, mathes_other

def run_single(model_path, image_file, output_path):

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
    model = model.eval().cuda().to(torch.bfloat16)

    # prompt = "<image>\nFree OCR. "
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    basename = Path(image_file).stem

    # infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

    # Tiny: base_size = 512, image_size = 512, crop_mode = False
    # Small: base_size = 640, image_size = 640, crop_mode = False
    # Base: base_size = 1024, image_size = 1024, crop_mode = False
    # Large: base_size = 1280, image_size = 1280, crop_mode = False

    # Gundam: base_size = 1024, image_size = 640, crop_mode = True

    # res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, save_results = True, test_compress = True)
    res = model.infer(tokenizer, prompt=prompt, image_file=image_file, 
                      output_path = output_path, base_size = 1024, image_size = 1024, 
                      crop_mode=False, save_results = True, eval_mode = True)

    content = clean_formula(res)
    matches_ref, mathes_other = re_match(content)
    for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
        content = content.replace(a_match_other, '').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').replace('<center>', '').replace('</center>', '')
    basename = Path(image_file).stem

    markdown_file = os.path.join(output_path, f"{basename}.md")
    with open(markdown_file, 'w', encoding='utf-8') as file:
        file.write(content)
        print(f"Saved: {markdown_file}")

def run_batch(model_path, input_dir, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
    model = model.eval().cuda().to(torch.bfloat16)

    prompt = "<image>\n<|grounding|>Convert the document to markdown."

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

    for root, _, files in os.walk(input_dir):
        for name in files:
            if any(name.lower().endswith(ext) for ext in image_extensions):
                img_path = os.path.join(root, name)
                basename = os.path.splitext(name)[0]
                markdown_file = os.path.join(output_dir, f"{basename}.md")

                if os.path.exists(markdown_file):
                    print(f"Already exist: {markdown_file}")
                    continue

                res = model.infer(tokenizer, prompt=prompt, image_file=img_path, 
                    output_path = output_dir, base_size = 1024, image_size = 1024, 
                    crop_mode=False, save_results=False, eval_mode=True)

                content = clean_formula(res)
                matches_ref, mathes_other = re_match(content)
                for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
                    content = content.replace(a_match_other, '').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').replace('<center>', '').replace('</center>', '')
                basename = Path(image_file).stem

                markdown_file = os.path.join(output_path, f"{basename}.md")
                with open(markdown_file, 'w', encoding='utf-8') as file:
                    file.write(content)
                    print(f"Saved: {markdown_file}")

# run single img
model_path = '../DeepSeek-OCR-ckpt'
image_file = 'your_image_file_path.jpg'
output_path = 'your_output_directory'
run_single(model_path, image_file, output_path)

# # run batch imgs
# model_path = '../DeepSeek-OCR-ckpt'
# image_file = 'your_image_file_path'
# output_path = 'your_output_directory'
# run_batch(model_path, input_dir, output_dir)