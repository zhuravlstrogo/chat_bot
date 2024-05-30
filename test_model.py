from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from datetime import datetime

start = datetime.now()
print(f"start test model at {start}")


# model_name = "under-tree/YAGPT"
# model_name = "microsoft/DialoGPT-medium"
# model_name = "IlyaGusev/saiga_llama3_8b" # нужна оперативка max 45 гб и 1 видюха, 57,5 k
# model_name = "IlyaGusev/saiga_llama3_8b_gguf"  # needs config.json, 17,8 k
# model_name = "IlyaGusev/saiga2_7b_lora" # needs config.json, example in extraction_saiga 
# model_name = 'Vikhrmodels/Vikhr-7b-0.1' # большая  
# model_name = 'ai-forever/ruGPT-3.5-13B' # 2,71 k, очень жирная! yandex 
# model_name = 'ai-forever/ruRoberta-large' # 101,378 k  try it! висит с оперативой 16 к
# https://github.com/yandex/YaLM-100B

# model_path = 'models/saiga_llama3_8b/' #  рабочая
# model_path = 'models/rubert-base-cased/' # try it! 
# model_path = 'models/saiga2_13b_lora/' # try it! 

# model_path = 'models/distilrubert-tiny-cased-conversational-v1' # ошибка 
# model_path = 'models/rubert-base-cased'

print(f'model_name {model_name}')

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# text = "why do you think that I will fall?"
text = "как дела?"

input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
# input_ids = tokenizer.encode(text, return_tensors="pt")

# chat_history_ids = model.generate(
# input_ids,
#  bos_token_id= 1,
#   do_sample= True,
#   eos_token_id= 2,
#   max_new_tokens= 3584,
#   no_repeat_ngram_size= 15,
#   pad_token_id= 0,
#   repetition_penalty= 1.2,
#   temperature= 0.5,
#   top_k= 30,
#   top_p= 0.9
# )


chat_history_ids = model.generate(
input_ids,
pad_token_id=tokenizer.eos_token_id,
max_length=100, # влияет на скорость работы 
max_new_tokens=100, # влияет на скорость работы 
do_sample=True,
# num_beams=5, # beam-search
)

output = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
print(f"DialoGPT: {output}")

# # # # # # # next version 
# model.eval()


# # util function to get expected len after tokenizing
# def get_length_param(text: str, tokenizer) -> str:
#     tokens_count = len(tokenizer.encode(text))
#     if tokens_count <= 15:
#         len_param = '1'
#     elif tokens_count <= 50:
#         len_param = '2'
#     elif tokens_count <= 256:
#         len_param = '3'
#     else:
#         len_param = '-'
#     return len_param


# # util function to get next person number (1/0) for Machine or Human in the dialogue
# def get_user_param(text: dict, machine_name_in_chat: str) -> str:
#     if text['from'] == machine_name_in_chat:
#         return '1'  # machine
#     else:
#         return '0'  # human


# chat_history_ids = torch.zeros((1, 0), dtype=torch.int)

# while True:
    
#     # next_who = input("Who's phrase?\t")  #input("H / G?")     # Human or GPT

#     # In case Human
#     # if next_who == "H" or next_who == "Human":
#     input_user = input("===> Human: ")
    
#     # encode the new user input, add parameters and return a tensor in Pytorch
#     new_user_input_ids = tokenizer.encode(f"|0|{get_length_param(input_user, tokenizer)}|" \
#                                             + input_user + tokenizer.eos_token, return_tensors="pt")

#     # append the new user input tokens to the chat history
#     chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

#     # if next_who == "G" or next_who == "GPT":

#     # next_len = input("Phrase len? 1/2/3/-\t")  #input("Exp. len?(-/1/2/3): ")
#     next_len = 2
#     # encode the new user input, add parameters and return a tensor in Pytorch
#     new_user_input_ids = tokenizer.encode(f"|1|{next_len}|", return_tensors="pt")
  
#     # append the new user input tokens to the chat history
#     chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    
#     # print(tokenizer.decode(chat_history_ids[-1])) # uncomment to see full gpt input
    
#     # save previous len
#     input_len = chat_history_ids.shape[-1]
#     # generated a response; PS you can read about the parameters at hf.co/blog/how-to-generate
#     chat_history_ids = model.generate(
#         chat_history_ids,
#         # num_return_sequences=1,                     # use for more variants, but have to print [i]
#         # max_length=512,
#         max_new_tokens=1000,
#         no_repeat_ngram_size=3,
#         do_sample=True,
#         # top_k=50,
#         # top_p=0.9,
#         # temperature = 0.6,                          # 0 for greedy
#         ## mask_token_id=tokenizer.mask_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#         ## unk_token_id=tokenizer.unk_token_id,
#         pad_token_id=tokenizer.pad_token_id,
#         ## device='cpu'
#     )
    
    
#     # pretty print last ouput tokens from bot
#     print(f"===> GPT-3:  {tokenizer.decode(chat_history_ids[:, input_len:][0], skip_special_tokens=True)}")

print(f'Pipeline worked {datetime.now() - start} seconds')