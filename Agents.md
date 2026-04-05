# Project Overview
I'm an undergraduate student working on a research project in the field of MLLM video question answering. This project is part of my research work.

I'm currently investigating how open source MLLMs handle unanswerable questions in video question answering tasks. I want to see if they can identify when a question cannot be answered based on the video content.


# Coding and Testing Guidelines
I believe most of the code will be python scripts.
I cannot run the code on my local machine, I will be testing on my universities high performance computing cluster. You can find detials about it here [HPC Cluster](https://docs.rcd.clemson.edu/palmetto/).

Do not execute code or create a conda environment on my local machine. I will be doing all the testing on the HPC cluster.

I am not too concerned about the code style, but I would like to have some comments in the code to help me understand it better.

I will be making subfolders for each experiment, so please make sure to put the code for each experiment in the correct subfolder.

Leave me instructions for setting up a conda environment on the HPC cluster, and make sure to include any necessary dependencies in the environment.yaml file.

Additionally Leave instructions for running any python scripts you write 

# Experiment Details
## Experiment 1: 
### Back ground 
For our first experiment, we will investigate the EgoBlind paper.
Link to the paper: [EgoBlind](https://arxiv.org/pdf/2503.08221)
Link to github repo: [EgoBlind GitHub](https://github.com/doc-doc/EgoBlind)


Heres some quick facts about the paper: 


EgoBlind (NeurIPS 2025, Datasets & Benchmarks Track) contains:
- **1,329 video clips** (~40s average) filmed from a first-person perspective by blind/visually impaired individuals
- **5,311 questions** posed or verified by blind users across 6 categories: Tool Use, Information Reading, Navigation, Safety Warnings, Social Communication, Other Resources
- **1–4 ground-truth answers per question** (avg. 3), annotated by sighted university students
- A deliberate **unanswerable subset** (~10.7% of answers are "I do not know") caused by poor visual quality, out-of-frame content, or the online QA task setting (answer not visible at question timestamp)

The paper intentionally keeps unanswerable questions *"to evaluate if models can reject to answer rather than hallucinate potentially malicious answers."* For a blind user relying on an AI assistant, a confidently wrong answer about an obstacle, direction, or safety hazard is dangerous. A model that hallucinates instead of saying "I don't know" represents a critical failure mode.

The paper uses **GPT-4o mini as a judge** — comparing each model's prediction against all 4 ground-truth answers using semantic similarity. A prediction is correct if it matches *any one* of the reference answers. The judge is prompted to allow paraphrasing and synonyms.

- **GPT Score**: % of predictions judged semantically correct
- **Human-AI agreement**: Cohen's Kappa = 0.73 (substantial), 88% of samples rated identically

### Key Paper Results (Overall Accuracy)

| Model | Overall Acc. |
|---|---|
| Human | 87.4% |
| GPT-4o | 59.3% |
| Gemini 2.5 Flash | 56.0% |
| InternVL2.5-26B | 55.0% |
| InternVL2.5-8B | 53.5% |
| LLaVA-Video | 51.5% |
| Gemini 1.5 Flash | 51.8% |
| Video-LLaMA3 | 49.2% |
| LLaVA-OV | 54.5% |
| Qwen2.5-VL | 45.5% |
| Video-LLaVA | 38.1% |
| LLaMA-VID | 39.1% |
| ShareGPT4Video | 32.9% |
| CogVLM2-Video | 40.3% |
| VILA1.5 | 48.2% |

**Note**: GPT-4o processes video via sampled frames (not native video upload). Gemini models accept native video files. All open-source models receive uniformly sampled frames up to the question timestamp.


### Experiment Details
**Research Question**: On questions that humans marked as unanswerable ("I don't know"), do open-source models correctly abstain, or do they hallucinate answers?

**HPC details**
the videos are on the HPC cluster as 
/scratch/jjtribb/EgoBlind_Videos/
they are orgnized badly 
cwd/split_0/00989.mp4 for example is video 989 in the test set. 

if you could write a one use script or command  to sort all the .mp4 file into a single folder that would be great. 


*From the paper (open-source):*
- `DAMO-NLP-SG/VideoLLaMA3-7B`
- `OpenGVLab/InternVL2_5-8B`
- `lmms-lab/llava-onevision-qwen2-7b-ov`
- `Qwen/Qwen2.5-VL-7B-Instruct`

*New models (not in the paper):*
we will get to this later.

**Output Metrics**:
- IDK Rate per model (% of unanswerable Qs answered with "I don't know")
- Hallucination Rate per model (% confidently answered despite GT being IDK)
- Comparison: Paper models vs. New models
- Breakdown by question type (Navigation, Safety, etc.)

