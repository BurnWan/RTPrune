# RTPrune: Reading-Twice Inspired Token Pruning for Efficient DeepSeek-OCR Inference

RTPrune is a two-stage, training-free visual token pruning framework for DeepSeek-OCR that mimics the LLM’s reading-twice behavior and adopts a dynamic pruning ratio, achieving efficient inference while preserving high OCR accuracy.

<p align="center" width="100%">
<img src="assets/dpsk-framework-1.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

## Highlights

<p align="center">
  <img src="assets/intro_accuracy-1.png" width="45%">
  <img src="assets/intro_efficiency-1.png" width="40%">
</p>

1. Our RTPrune consistently outperforms prior token pruning methods on DeepSeek-OCR, retaining over 97.88\% of accuracy with 84\% of visual tokens on olmOCR-Bench. 
2. Our RTPrune reduces GFLOPs by nearly 15.29\% and prefilling time by nearly 21.26\% on OmniDocBench when maintaining 99.47\% accuracy.

## Installation

1. Install the [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) environment.

2. download the ckpt files from [huggingface](https://huggingface.co/deepseek-ai/DeepSeek-OCR) and put them in ./DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-ckpt

3. replace or add the corresponding files with our [code](./DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-ckpt) and the added part can be searched by "\[modified\]"

## Quick Start

run the following command:
```
cd dpsk-ocr-token-pruning/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-hf
bash run_dpsk_ocr.py
```

We also provide the implementation of [VisionZip](https://github.com/dvlab-research/VisionZip), [DivPrune](https://github.com/vbdi/divprune) and [CDPruner](https://github.com/Theia-4869/CDPruner) on DeepSeek-OCR


## Evaluation

The evaluation code follows the pipeline of [OmniDocBench](https://github.com/opendatalab/OmniDocBench?tab=readme-ov-file), [olmOCR-Bench](https://github.com/allenai/olmocr/tree/main/olmocr/bench) and [Ocean-OCR Benchmark](https://github.com/guoxy25/Ocean-OCR?tab=readme-ov-file).

## Acknowledgement
- This work is built upon [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR). We thank them for their excellent open-source contributions.

- We also thank [VisionZip](https://github.com/dvlab-research/VisionZip), [DivPrune](https://github.com/vbdi/divprune), [CDPruner](https://github.com/Theia-4869/CDPruner), and others for their contributions, which have provided valuable insights.

## License
- RTPrune is licensed under the Apache License 2.0. 