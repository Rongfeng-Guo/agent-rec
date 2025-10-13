# Optimizing Multi-Turn Interactive Recommendation Agents via Generative Intrinsic Motivation

**The code is being cleaned up and will be finalized before October 20.**

![alt text](pic/image.png)

## Usage
### Getting Start
You can use following scripts to install related python package through pip:
```
git clone https://github.com/XueyangFeng/GIMO.git
cd GIMO
pip install -r requirements.txt
```


### **AILO Environment Setup**

Following ECPO, We provide a detailed pipeline for the AILO environment, including additional [README files](./user_simulator/readme.md). For a quick setup, follow these steps:

1. Download the [index file](https://drive.google.com/file/d/1P6QkUrikHnwxNov0fUY3SxWQkl1qve0O/view?usp=drive_link).
2. Unzip the downloaded file into the `user_simulator/embedding/` folder.

### **API Configuration**
All LLM (Large Language Model) calls in this repository are made using OpenAI-like interfaces. To configure the APIs:

1. Set your API information in the `config/api_config.json` file.
2. For closed-source models, set the information directly in the config.
3. For open-source models, use `vllm` for local deployment. We have provided an example script in the `model/` directory.


### **Running GIMO**

To run the existing prompt-based Conversational Recommendation Agent (CRA) or an aligned CRA, you can set the relevant configuration in the `main.sh` file and execute it.

Our CRA alignment process consists of four main stages:
1. **SFT (Stage 1)**: Supervised Fine-Tuning & Cold Start
2. **ECPO (Stages 2-4)**: Generative Intrinsic Motivation based Optimization

### **Stages Overview:**
- **SGPT (Stage 1)**: [Supervised Fine-Tuning & Cold Start](todo)
- **ECPO Stages (2-4)**:
  - [Generative Potential Estimation](todo)
  - [Hint-conditioned Action Proposal](todo)
  - [Conditional Direct Preference Optimization](todo)


### Evaluation

Test recommendation metric using simulator environment:
```
# test the existing prompt-based CRA baseline
bash main.sh
# test the aligned CRA
bash main_lora.sh
```


## References
1. Our evaluation method is based on [XueyangFeng/ECPO](https://github.com/XueyangFeng/ECPO).
2. Our training code is based on [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
