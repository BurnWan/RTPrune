from transformers import AutoModel, AutoTokenizer
import torch
import os
import re
from PIL import Image
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    # pattern1 = r'<\|ref\|>.*?<\|/ref\|>\n'
    # new_text1 = re.sub(pattern1, '', text, flags=re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other
def deepseek_ocr_post_process(res):
    
    outputs = res.strip()

    matches_ref, matches_images, mathes_other = re_match(outputs)

    for idx, a_match_image in enumerate(matches_images):
        outputs = outputs.replace(a_match_image, '![](images/' + str(idx) + '.jpg)\n')

    for idx, a_match_other in enumerate(mathes_other):
        outputs = outputs.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
    return outputs

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
    res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 1024, crop_mode=False, save_results = True, eval_mode = True)

    outputs = deepseek_ocr_post_process(res)
    markdown_file = os.path.join(output_path, f"{basename}.md")
    with open(markdown_file, 'w', encoding='utf-8') as file:
        file.write(outputs)
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
                    print(f"文件已存在，跳过: {markdown_file}")
                    continue

                res = model.infer(tokenizer, prompt=prompt, image_file=img_path, 
                    output_path = output_dir, base_size = 1024, image_size = 1024, 
                    crop_mode=False, save_results=False, eval_mode=True)

                outputs = deepseek_ocr_post_process(res)
                markdown_file = os.path.join(output_dir, f"{basename}.md")

                with open(markdown_file, 'w', encoding='utf-8') as file:
                    file.write(outputs)
                    print(f"Saved: {markdown_file}")

# # run single img
# model_path = '/export/home/wanben.burn/github/dpsk-ocr-token-pruning/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-ckpt'
# # image_file = 'your_image.jpg'
# # output_path = 'your/output/dir'
# run_single(model_path, image_file, output_path)

# run batch imgs
model_path = '/export/home/wanben.burn/github/dpsk-ocr-token-pruning/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-ckpt'
input_dir = '/export/home/wanben.burn/OmniDocBench/data/image_samples'
output_dir = '/export/home/wanben.burn/github/dpsk-ocr-token-pruning/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-hf/output/dpskocr_base_sample/md'
run_batch(model_path, input_dir, output_dir)