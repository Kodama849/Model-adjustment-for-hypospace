# Enhancing Large Language Model Performance on the HypoSpace Benchmark

## Project Context
This project is based on the [HypoSpace](https://github.com/CTT-Pavilion/_HypoSpace) benchmark proposed by Chen *et al.* (2025), which evaluates large language models (LLMs) as **set-valued hypothesis generators** under conditions of **underdetermination** [@chen2025hypospace].  
The benchmark consists of three primary tasks designed to assess model creativity, reasoning robustness, and structural consistency across different cognitive domains.

## Project Objective
The goal of this project is to improve LLM performance across all three HypoSpace benchmark tasks by refining prompt engineering strategies and reasoning control mechanisms.  
The focus lies in increasing **Validity**, **Novelty**, and **Recovery** — key indicators of a model’s ability to generate logically sound, diverse, and reconstructable hypotheses.  
Any open or commercial LLM API could be used, allowing comparative analysis between reasoning-oriented and non-reasoning models.

## Usage Reminder
You have to input your api tokens in config file

## Citation
If you refer to or build upon the HypoSpace framework in your research or coursework, please cite the original paper:

```bibtex
@article{chen2025hypospace,
  title={HypoSpace: Evaluating LLM Creativity as Set-Valued Hypothesis Generators under Underdetermination},
  author={Chen, Tingting and Lin, Beibei and Yuan, Zifeng and Zou, Qiran and He, Hongyu and Ong, Yew-Soon and Goyal, Anirudh and Liu, Dianbo},
  journal={arXiv preprint arXiv:2510.15614},
  year={2025}
}
