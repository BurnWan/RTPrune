# RTPrune: Reading-Twice Inspired Token Pruning for Efficient DeepSeek-OCR Inference

RTPrune is a two-stage, training-free visual token pruning framework for DeepSeek-OCR that mimics the LLM’s reading-twice behavior and adopts a dynamic pruning ratio, achieving efficient inference while preserving high OCR accuracy.

<p align="center" width="100%">
<img src="assets/dpsk-framework-1.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>


## Usage
1. download the ckpt files from huggingface and put them in ./DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-ckpt
2. replace the corresponding files with ours