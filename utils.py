from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# TODO: нужна ли async?
# async def generate_text(prompt) -> dict:
#     try:
#         response = await openai.ChatCompletion.acreate(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "user", "content": prompt}
#             ]
#         )
#         return response['choices'][0]['message']['content'], response['usage']['total_tokens']
#     except Exception as e:
#         logging.error(e)
        


def generate_text(text):
    try:
        # model_name = "microsoft/DialoGPT-medium"
        model_name = "models/saiga_llama3_8b/"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
        # concatenate new user input with chat history (if there is)
        # bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids

        # TODO: запоминать контекст 
        chat_history_ids = torch.zeros((1, 0), dtype=torch.int)
        # generate a bot response
        chat_history_ids = model.generate(
        input_ids,
        max_length=100, # was 1000
        max_new_tokens=100, 
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
        )
        #print the output
        output = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"You: {text}")
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ' )
        print(f"DialoGPT: {output}")
        return output
    except Exception as e:
        print(f'{e}')
