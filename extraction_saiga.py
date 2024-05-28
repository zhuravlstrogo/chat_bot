import torch
# from langchain.document_loaders import DataFrameLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
import pandas as pd
from peft import PeftModel, PeftConfig
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch.nn.functional as F

from typing import Any, List, Mapping, Optional


# MODEL_NAME = "IlyaGusev/saiga_mistral_7b_lora"
# MODEL_NAME = "Gaivoronsky/Mistral-7B-Saiga"
# MODEL_NAME = "IlyaGusev/saiga2_7b_lora" # по дефолту в примере
MODEL_NAME = 'saiga_llama3_8b/' 
DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>"
DEFAULT_RESPONSE_TEMPLATE = "<s>bot\n"
# DEFAULT_SYSTEM_PROMPT = "Ты извлекаешь информацию из текста"
DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'DEVICE: {device}')


# https://www.kaggle.com/code/yarasslove/extraction-saiga

class Conversation:
    def __init__(
        self,
        message_template=DEFAULT_MESSAGE_TEMPLATE,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        start_token_id=1,
        bot_token_id=9225
    ):
        self.message_template = message_template
        self.start_token_id = start_token_id
        self.bot_token_id = bot_token_id
        self.messages = [{
            "role": "system",
            "content": system_prompt
        }]

    def get_start_token_id(self):
        return self.start_token_id

    def get_bot_token_id(self):
        return self.bot_token_id

    def add_user_message(self, message):
        self.messages.append({
            "role": "user",
            "content": message
        })

    def add_bot_message(self, message):
        self.messages.append({
            "role": "bot",
            "content": message
        })

    def get_prompt(self, tokenizer):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += tokenizer.decode([self.start_token_id, self.bot_token_id])
        return final_text.strip()



def generate(model, tokenizer, prompt, generation_config):
    print()
    data = tokenizer(prompt,
                     return_tensors="pt",
                     add_special_tokens=False,
    #                      padding=True,
    #                     truncation=True
                    )
    #print(data)
    data = {k: v.to(device) for k, v in data.items()}
    
    output_ids = model.generate(
        **data,
        generation_config = generation_config
    #         remove_invalid_values = True
    )[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=False)
    return output.strip()


import time

# st_time = time.time()
# config = PeftConfig.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(
#     config.base_model_name_or_path,
#     # load_in_8bit = True, # требует GPU для квантизации 
#     torch_dtype=torch.float16,
#     device_map="auto"
# )
# model = PeftModel.from_pretrained(
#     model,
#     MODEL_NAME,
#     torch_dtype=torch.float16,
#     is_trainable = True,
#     device_map="auto"
# )


# model.eval()

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
# generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
# print('generation_config')
# print(generation_config)


def get_message(inputs):
    conversation = Conversation()
    conversation.add_user_message(inputs)
    prompt = conversation.get_prompt(tokenizer)
    print('Промт', '\n', '*'*100)
    print(prompt)
    print('*'*100)
    output = generate(model, tokenizer, prompt, generation_config)
    return output


get_message('сколько в среднем весит человек')


print(f'Прошло времени {time.time() - st_time}')