<p align="center">
    <a href=""><img width="765" alt="image" src="assets/title.png"></a> 
</p>
<p align="center">
    <a href="https://haojunchen663.github.io/"><strong>Haojun Chen</strong></a><sup>1,*</sup>
    ,
    <a href="https://scholar.google.com/citations?user=8NfQv1sAAAAJ&hl=en"><strong>Haoyang He</strong></a><sup>1,*</sup>
    ,
    <a href="https://scholar.google.com/citations?user=pjcYzvYAAAAJ&hl=zh-CN&oi=ao"><strong>Chengming Xu</strong></a><sup>2,*</sup>
    ,
    <a href="https://scholar.google.com/citations?user=gUJWww0AAAAJ"><strong>Qingdong He</strong></a><sup>1</sup>
    ,
    <a href="https://scholar.google.com/citations?user=-OxQlHsAAAAJ&hl=en"><strong>Junwei Zhu</strong></a><sup>3</sup>
    ,
    <a href="https://scholar.google.com/citations?user=xiK4nFUAAAAJ&hl=en"><strong>Yabiao Wang</strong></a><sup>1</sup>
    ,
    <a href="https://xzc-zju.github.io/"><strong>Zhucun Xue</strong></a><sup>1</sup>
    ,
    <br><a href="https://scholar.google.com/citations?hl=zh-CN&user=tgDc0fsAAAAJ"><strong>Xianfang Zeng</strong></a><sup>1</sup>
    ,
    <a href="https://scholar.google.com/citations?user=edoqkgoAAAAJ&hl=en"><strong>Zhennan Chen</strong></a><sup>3</sup>
    ,
    <a href="https://scholar.google.com.hk/citations?user=3lMuodUAAAAJ&hl=zh-CN&oi=ao"><strong>Xiaobin Hu</strong></a><sup>4</sup>
    ,
    <a href="https://sites.google.com/view/fromandto"><strong>Hao Zhao</strong></a><sup>5</sup>
    ,
    <a href="https://person.zju.edu.cn/yongliu"><strong>Yong Liu</strong></a><sup>1</sup>
    ,
    <a href="https://zhangzjn.github.io/"><strong>Jiangning Zhang</strong></a><sup>1<a href="mailto:186368@zju.edu.cn">✉</a></sup>
    ,
    <a href="https://scholar.google.com/citations?user=RwlJNLcAAAAJ&hl=en"><strong>Dacheng Tao</strong></a><sup>6</sup>
</p>
<p align="center">
    <sup>1</sup><strong>Zhejiang University</strong> &nbsp;&nbsp;&nbsp; 
    <sup>2</sup><strong>Fudan University</strong> &nbsp;&nbsp;&nbsp; 
    <sup>3</sup><strong>Nanjing University</strong> &nbsp;&nbsp;&nbsp;
    <sup>4</sup><strong>National University of Singapore</strong>
    <br><sup>5</sup><strong>Tsinghua University</strong> &nbsp;&nbsp;&nbsp;
    <sup>6</sup><strong>Nanyang Technological University</strong>
</p>
<p align="center">
    <a href='https://arxiv.org/abs/2605.20147'>
      <img src='https://img.shields.io/badge/arXiv-PDF-red?style=flat&logo=arXiv&logoColor=red'
      alt='arXiv PDF'>
    </a> 
    <a href='https://haojunchen663.github.io/projects/PixVerve/'>
      <img src='https://img.shields.io/badge/PixVerve-Page-93edc7?style=flat&logo=googlechrome&logoColor=93edc7'
      alt='webpage-Web'>
    </a>
    <a href='https://huggingface.co/datasets/HaojunChen/PixVerve-95K'>
      <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Benchmark-ffcc4d">
    </a>
    <a href='https://modelscope.cn/datasets/APRIL6AIGC/PixVerve-95K'>
      <img src="https://img.shields.io/badge/ModelScope-Dataset | Benchmark-624aff?logo=modelscope" />
    </a>
</p>

## 🔥 News

- __[2026.05.20]__: We release the [paper](https://arxiv.org/abs/2605.20147), the [project page](https://haojunchen663.github.io/projects/PixVerve/), the [PixVerve-95K](https://modelscope.cn/datasets/APRIL6AIGC/PixVerve-95K) dataset, the [PixVerve-Bench](https://huggingface.co/datasets/HaojunChen/PixVerve-95K) benchmark, and the [github repo](https://github.com/HaojunChen663/PixVerve-95K).

<a name="introduction"></a>

## 📷 Introduction
💡**TL;DR:** 
[**PixVerve**](https://arxiv.org/abs/2605.20147) explores and proposes a comprehensive methodology framework spanning dataset, model, and benchmark, taking a pioneering step to advance native text-to-image generation to 100MP.

<a name="highlight"></a>

## ✨ Highlights
1. We introduce **PixVerve-95K**, the first large-scale, high-quality T2I dataset to push image resolution to 100MP. With a five-stage, automated data pipeline, we curate 95,735 100MP images with fine-grained annotations (5 types of metadata and 2 comprehensive captions), directly supporting the training or fine-tuning of T2I models at high resolutions.
2. Based on our proposed PixVerve-95K, we first **explore the attempt of generating 100MP images natively**. Specifically, we extend existing T2I foundation models (including both latent diffusion models and pixel diffusion models) with three distinct training schemes, providing valuable insights and paving the way for future breakthroughs.
3. To address the limitations of conventional T2I benchmarks, we construct **PixVerve-Bench**, a systematic, hierarchical evaluation protocol incorporating both traditional metrics and assessments based on Multimodal Large Language Models (MLLMs).

# :mailbox_with_mail:Summary of Contents

- [Introduction](##introduction)  
- [Highlights](##highlight)  