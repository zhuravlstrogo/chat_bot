from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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
        model_name = "microsoft/DialoGPT-medium"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        text = "I fell in love. what should I do?"
        # text = "привет"

        input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
        # concatenate new user input with chat history (if there is)
        # bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids
        bot_input_ids = input_ids
        # generate a bot response
        chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
        )
        #print the output
        output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"DialoGPT: {output}")
        return output
    except Exception as e:
        print(f'{e}')
