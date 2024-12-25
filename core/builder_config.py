"""Configuration."""
import streamlit as st
import os

### DEFINE BUILDER_LLM #####
## Uncomment the LLM you want to use to construct the meta agent

# ## OpenAI
# from llama_index.llms.openai import OpenAI

# # set OpenAI Key - use Streamlit secrets
# os.environ["OPENAI_API_KEY"] = st.secrets.openai_key # check model at QuestionAnswering.py (Quang)
# # load LLM
# BUILDER_LLM = OpenAI(model="gpt-3.5-turbo")

# from llama_index.llms.ollama import Ollama
# from llama_index.core import Settings

# Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)
# BUILDER_LLM = Settings.llm

os.environ["GOOGLE_API_KEY"] = st.secrets.gemini_key

from llama_index.llms.gemini import Gemini
from llama_index.core.settings import Settings

Settings.llm = Gemini(model="models/gemini-1.5-flash")
BUILDER_LLM = Settings.llm

# from llama_index.llms.huggingface import HuggingFaceLLM
# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Chọn tokenizer cho model của bạn

# # Định nghĩa template Jinja
# chat_template = """
# {%- for message in messages %}
#     {{- message['role'] }}: {{ message['content'] }}
# {%- endfor %}
# {%- if add_generation_prompt %}
#     Assistant: 
# {%- endif %}
# """

# # Tạo danh sách tin nhắn
# messages = [
#     {"role": "user", "content": "Xin chào!"},
#     {"role": "assistant", "content": "Chào bạn!"},
# ]

# # Áp dụng template và tokenize
# tokenized_chat = tokenizer.apply_chat_template(
#     messages, chat_template=chat_template, tokenize=True, add_generation_prompt=True
# )

# # remotely_run = HuggingFaceInferenceAPI(
# #     model_name="HuggingFaceH4/zephyr-7b-alpha", token=st.secrets.huggingface_key
# # )
# locally_run = HuggingFaceLLM(model_name="openai-community/gpt2", tokenizer=tokenizer)
# BUILDER_LLM = locally_run

# # Anthropic (make sure you `pip install anthropic`)
# from llama_index.core import Settings
# from llama_index.llms.anthropic import Anthropic
# # set Anthropic key
# os.environ["ANTHROPIC_API_KEY"] = st.secrets.anthropic_key
# Settings.llm = Anthropic(model="claude-3-opus-20240229")
# BUILDER_LLM = Settings.llm
