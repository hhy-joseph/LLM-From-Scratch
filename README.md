# Machine Learning from Scratch

This is a collection of personal projects aimed at building machine learning models from the ground up, without relying on predefined packages or high-level abstractions. Each project is an exercise in understanding core ML concepts through hands-on implementation, prioritizing learning over speed or optimization.

## Projects

### Project 1: Transformer from Scratch

**Repository/Section:** *Transformer-from-Scratch*

**Goal:** Implement the Transformer architecture as outlined in *"Attention is All You Need"* (Vaswani et al., 2017).

**Tools:** NumPy and PyTorch

**Details:**
- A complete, custom-built Transformer, including multi-head self-attention, positional encodings, and feed-forward networks.
- When using PyTorch, only basic tensor operations are employed—no predefined nn.Transformer, nn.Embedding, or similar classes.
- Focus is on dissecting and understanding the Transformer's mechanics, not on performance or scalability.

### Project 2: LLM from Scratch

**Repository/Section:** *LLM-from-Scratch*

**Goal:** Build a simple Large Language Model (LLM) from the ground up.

**Tools:** PyTorch

**Details:**
- A fully custom implementation, including tokenization, attention layers, and training procedures.
- No use of prebuilt libraries for embeddings, transformers, or optimization—everything is coded manually.
- Aimed at exploring the fundamentals of language model design and training through practical experimentation.

### Project 3: Coming Soon

**Details:** Additional projects are in the planning phase. More information will be added as new ideas take shape.

## Purpose

These projects are personal learning exercises to solidify understanding of machine learning concepts by implementing them at a low level. They are not intended for production use or optimized performance but serve as a playground for studying ML theory and practice.

## Setup

This project uses uv for package management to handle dependencies like NumPy and PyTorch.

To set up the environment:

1. Install uv if you haven't already:
   ```bash
   pip install uv
   ```

2. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

3. Sync the dependencies using uv:
   ```bash
   uv sync
   ```

This ensures a lightweight and reproducible environment tailored to the project's needs.

## License

These projects are for personal education and experimentation. They are not currently intended for commercial use or broad distribution.

## Contact

For questions, suggestions, or feedback, feel free to reach out via [insert contact method, e.g., GitHub issues or email].

### Suggested Repository Names

**Single Portfolio Repo:**
- ML-From-Scratch-Portfolio
- MachineLearningGroundUp
- DIY-ML-Experiments
- CoreMLImplementations

**Separate Repos:**
- Transformer: Transformer-from-Scratch
- LLM: LLM-from-Scratch