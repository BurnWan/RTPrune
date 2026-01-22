# RTPrune: Reading-Twice Inspired Token Pruning for Efficient DeepSeek-OCR Inference

RTPrune is a two-stage, training-free visual token pruning framework for DeepSeek-OCR that mimics the LLM’s reading-twice behavior and adopts a dynamic pruning ratio, achieving efficient inference while preserving high OCR accuracy.

<p align="center" width="100%">
<img src="assets/dpsk-framework-1.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

## Highlights
<p align="center">
  <img src="assets/intro_accuracy-1.png" width="30%">
  <img src="assets/intro_efficiency-1.png" width="27%">
</p>

1. Our RTPrune consistently outperforms prior token pruning methods on DeepSeek-OCR, retaining over 97.88\% of accuracy with 84\% of visual tokens on olmOCR-Bench. 
2. Our RTPrune reduces GFLOPs by nearly 15.29\% and prefilling time by nearly 21.26\% on OmniDocBench when maintaining 99.47\% accuracy.

## Usage
1. download the ckpt files from huggingface and put them in ./DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-ckpt
2. replace the corresponding files with ours