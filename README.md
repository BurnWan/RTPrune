# RTPrune: Reading-Twice Inspired Token Pruning for Efficient DeepSeek-OCR Inference

## 🔥News
- [2026.05.01] 🎉 Our training-free inference acceleration method [RTPrune](https://arxiv.org/abs/2605.00392) has been accepted at **ICML 2026**.

## ✨Highlights

<p align="center">
  <img src="assets/intro_accuracy-1.png" width="45%">
  <img src="assets/intro_efficiency-1.png" width="40%">
</p>

1. Our RTPrune consistently outperforms prior token pruning methods on DeepSeek-OCR, retaining over 97.88\% of accuracy with 84\% of visual tokens on olmOCR-Bench. 
2. Our RTPrune reduces GFLOPs by nearly 15.29\% and prefill time by nearly 18.90\% on OmniDocBench when maintaining 99.47\% accuracy.

## 🌈Method

<p align="center" width="100%">
<img src="assets/dpsk-framework-1.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

1. We introduce RTPrune, a plug-and-play visual token pruning method in DeepSeek-OCR which mimics the reading twice behavior of the LLM via a two-stage pipeline: retaining high-norm tokens and then merging the remaining ones via optimal transport.
2. We propose a dynamic pruning strategy to enable a better efficiency–accuracy trade-off, which combines the post-encoding inter-token similarity and the original textual density of the image.

## 📦Installation

1. Install the [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) environment.

2. Download the ckpt files from [huggingface](https://huggingface.co/deepseek-ai/DeepSeek-OCR) and put them in ./DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-ckpt.

3. Replace the corresponding files or add new files with our [code](./DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-ckpt) and the added part can be searched by "\[modified\]".

## 🚀Quick Start

Run the following command:
```
cd DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr.py
```


## 📊Evaluation

- The evaluation code follows the pipeline of [OmniDocBench](https://github.com/opendatalab/OmniDocBench?tab=readme-ov-file), [olmOCR-Bench](https://github.com/allenai/olmocr/tree/main/olmocr/bench) and [Ocean-OCR Benchmark](https://github.com/guoxy25/Ocean-OCR?tab=readme-ov-file).

- The evaluation for prefilling time and decoding time is provided in our [code](./DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-ckpt/modeling_deepseekv2.py).

- We also provide the implementations of [VisionZip](https://github.com/dvlab-research/VisionZip), [DivPrune](https://github.com/vbdi/divprune) and [CDPruner](https://github.com/Theia-4869/CDPruner) on DeepSeek-OCR.

## 👏Acknowledgement
- This work is built upon [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR). We thank them for their excellent open-source contributions.

- We also thank [VisionZip](https://github.com/dvlab-research/VisionZip), [DivPrune](https://github.com/vbdi/divprune), [CDPruner](https://github.com/Theia-4869/CDPruner), and others for their contributions, which have provided valuable insights.

<!-- ## 📜Citation

If you find this project useful in your research, please consider citing:

```bib
@inproceedings{
anonymous2026rtprune,
title={{RTP}rune: Reading-Twice Inspired Token Pruning for Efficient DeepSeek-{OCR} Inference},
author={Ben Wan, Yan Feng, Zihan Tang, Weizhe Huang, Yuting Zeng, Jia Wang and Tongxuan Liu},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=bniyv9QWYc}
}
``` -->