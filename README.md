# 🧠 LangChain + OpenAI OSS 120B Integration

This project integrates [OpenAI's OSS 120B](https://huggingface.co/openai/gpt-oss-120b) model with [LangChain](https://www.langchain.com/) using the HuggingFace Transformers pipeline.

It demonstrates how to:
- Load the open-source `gpt-oss-120b` model
- Wrap it using LangChain's `HuggingFacePipeline` LLM wrapper
- Run prompt-based chains
- Add conversational memory using LangChain

---

## 🔧 Setup Instructions

### ✅ Requirements

Install the dependencies:

```bash
pip install -U transformers torch langchain
