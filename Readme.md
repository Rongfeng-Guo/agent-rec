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

To run the existing prompt-based Conversational Recommendation Agent (IRA) or an aligned IRA, you can set the relevant configuration in the `main.sh` file and execute it.

Our IRA alignment process consists of four main stages:
1. **SFT (Stage 1)**: Supervised Fine-Tuning & Cold Start
2. **GIMO (Stages 2-4)**: Generative Intrinsic Motivation based Optimization

### **Stages Overview:**
- **SFT & Cold Start (Stage 1)**: [Supervised Fine-Tuning & Cold Start](todo)
- **GIMO Stages (2-4)**:
  - [Generative Potential Estimation](todo)
  - [Hint-conditioned Action Proposal](todo)
  - [Conditional Direct Preference Optimization](todo)



#### ⚙️ Implementation Integration

We have **seamlessly integrated** CDPO into the **LLaMA-Factory** training framework.  
No additional setup is required.

The core implementation can be found at:
```
LLaMA-Factory/src/llamafactory/train/dpo/train.py
```
The key implementation extends standard DPO loss with an action-level KL regularization term to achieve conditional alignment:

```python
def adpo_loss(
    self,
    chosen_logps: torch.FloatTensor,
    rejected_logps: torch.FloatTensor,
    ref_chosen_logps: torch.FloatTensor,
    ref_rejected_logps: torch.FloatTensor,
    action_policy_logps: torch.FloatTensor,     # [B, T, V]
    action_reference_logps: torch.FloatTensor,  # [B, T, V]
    action_mask: torch.BoolTensor,              # [B, T, 1]
    kl_type: str = "l2",
    kl_coef: float = 1.0,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

    # === Step 1: Standard DPO loss ===
    dpo_losses, dpo_chosen_rewards, dpo_rejected_rewards = self.dpo_loss(
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps
    )

    # === Step 2: Action-level KL regularization (masked) ===
    if kl_type == "l2":
        diff = (action_policy_logps - action_reference_logps) ** 2
    elif kl_type == "abs":
        diff = torch.abs(action_policy_logps - action_reference_logps)
    else:
        raise ValueError(f"Unsupported kl_type: {kl_type}")

    # Masked KL only on valid action regions
    masked_diff = diff * action_mask  # [B, T, V]
    vocab_size = diff.size(-1)
    token_count = action_mask.sum(dim=(1, 2)).clamp(min=1)

    kl_reg = masked_diff.sum(dim=(1, 2)) / (token_count * vocab_size)

    # === Step 3: Weighted combination ===
    total_loss = dpo_losses + kl_coef * kl_reg

    return total_loss, dpo_chosen_rewards, dpo_rejected_rewards
```

A ready-to-run CDPO training script is provided in the LLaMA-Factory repository.

```
cd LLaMA-Factory
bash gimo/{dataset}/gimo/adpo_v1_sample1.sh
```


### Evaluation

Test recommendation metric using simulator environment:
```
# test the existing prompt-based IRA baseline
bash main.sh
# test the trained IRA
bash main_lora.sh
```


## References
1. Our evaluation method is based on [XueyangFeng/ECPO](https://github.com/XueyangFeng/ECPO).
2. Our training code is based on [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).
