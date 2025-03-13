# Training Mistral-7B-Instruct on Databricks Dolly-15k with Torch, RAY, and vLLM

## Overview

This project focuses on training the **Mistral-7B-Instruct** model using the **Databricks Dolly-15k** QA dataset. The goal is to fine-tune the model for better instruction-following capabilities. We employ **Torch** for deep learning, **RAY** for distributed computing, and **vLLM**'s **ROUGE function** for evaluation. **DeepSpeed** is used to enhance GPU-based training efficiency.

## Tech Stack Used
- **Programming Language:** Python
- **Deep Learning Framework:** PyTorch
- **Distributed Training:** RAY Train, DeepSpeed
- **Evaluation:** vLLM (ROUGE metric)
- **Dataset Handling:** Hugging Face Datasets
- **Model Serving:** Transformers

## Prerequisites
Before running the project, ensure you have the following:
- A machine with **multiple GPUs** (recommended for distributed training)
- **Python 3.8+** installed
- **CUDA-enabled GPU** with appropriate drivers installed
- **pip** package manager updated
- Basic knowledge of deep learning and model fine-tuning

## Model and Dataset

- **Model:** [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct), an instruction-tuned variant of Mistral-7B.
- **Dataset:** [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), a human-generated instruction-following dataset.

## Installation

To set up the environment, install the required dependencies:

```bash
pip install torch ray[train] vllm deepspeed transformers datasets rouge-score
```

## Repository Structure

```
├── data/                                                # Dataset storage
├── scripts/                                             # Training and evaluation scripts
│   ├── Mistral-instruct-withRay-vLLM.ipynb              # Main training script
├── models/                                              # Trained model checkpoints
├── configs/                                             # Configuration files for training
│   ├── ds_config.json                                   # DeepSpeed configuration file
├── README.md                                            # Project documentation
└── requirements.txt                                     # List of dependencies
```

## Tools and Their Purpose

### **Torch**

**Purpose:**

- Torch provides a robust framework for deep learning, enabling efficient training and inference. It offers automatic differentiation and GPU acceleration, making it ideal for handling large-scale transformer models like Mistral-7B-Instruct.

**Advantages:**

- Provides flexible and efficient GPU acceleration.
- Seamless integration with transformers and deep learning libraries.
- Supports automatic differentiation and model optimization.

### **RAY Train**

**Purpose:**

- RAY Train enables large-scale distributed training by efficiently managing parallel computations across multiple GPUs or nodes. It simplifies workload balancing and fault tolerance to ensure seamless execution.

**Advantages:**

- Simplifies parallel training and workload distribution.
- Provides fault tolerance and automatic workload balancing.
- Scales effortlessly for large model training.

### **vLLM**

**Purpose:**

- vLLM is used for evaluation, providing an efficient way to measure generated responses against ground-truth answers. It leverages the ROUGE metric to assess text quality and similarity.

**Advantages:**

- Optimized for efficient large-scale inference.
- Provides high-speed token generation.
- Enables more accurate and scalable evaluation.

### **DeepSpeed**

**Purpose:**

- DeepSpeed optimizes memory usage and computational efficiency for training large-scale models. It utilizes ZeRO optimization to enable effective distributed training across GPUs without excessive resource consumption.

**Advantages:**

- Supports ZeRO (Zero Redundancy Optimizer) for reduced memory consumption.
- Enhances distributed training performance with efficient data parallelism.
- Enables training of large models without requiring excessive GPU memory.


## Conclusion

This setup effectively fine-tunes **Mistral-7B-Instruct** on **Databricks Dolly-15k** while optimizing performance using **DeepSpeed, RAY, and vLLM**. The combination of these tools ensures scalable, efficient distributed training and robust evaluation using **ROUGE metrics**.

## Citations

**Mistral-7B-Instruct:**

```
@article{jiang2023mistral,
  title={Mistral-7B: A Strong Open-Weight Language Model},
  author={Jiang et al.},
  year={2023}
}
```

**Databricks Dolly 15k:**

```
@article{databricks2023dolly,
  title={Databricks Dolly 15k: High-Quality Human-Generated Instruction-Following Dataset},
  author={Databricks},
  year={2023}
}
```

