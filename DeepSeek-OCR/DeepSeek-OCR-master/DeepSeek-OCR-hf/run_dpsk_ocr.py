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

model_name = 'deepseek-ai/DeepSeek-OCR'


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)



# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'your_image.jpg'
output_path = 'your/output/dir'
basename = Path(image_file).stem



# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True

# res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, save_results = True, test_compress = True)
res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 1024, crop_mode=False, save_results = True, test_compress = True)

outputs = deepseek_ocr_post_process(res)
markdown_file = os.path.join(output_path, f"{basename}.md")
with open(markdown_file, 'w', encoding='utf-8') as file:
    file.write(outputs)
    print(f"Saved: {markdown_file}")