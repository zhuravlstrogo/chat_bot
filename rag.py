import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = 'IlyaGusev/saiga_llama3_8b'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype = torch.bfloat16,
    device_map = "auto"
)

ef get_llm_answer(query, chunks_join, max_new_tokens, temperature, top_p, top_k):
    user_prompt = '''Используй только следующий контекст, чтобы очень кратко ответить на вопрос в конце.
    Не пытайся выдумывать ответ.
    Контекст:
    ===========
    {chunks_join}
    ===========
    Вопрос:
    ===========
    {query}'''.format(chunks_join=chunks_join, query=query)
    
    SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
    RESPONSE_TEMPLATE = "<|im_start|>assistant\n"
    
    prompt = f'''<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n{RESPONSE_TEMPLATE}'''
    
    def generate(model, tokenizer, prompt):
        data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        data = {k: v.to(model.device) for k, v in data.items()}
        output_ids = model.generate(
            **data,
            bos_token_id=128000,
            eos_token_id=128001,
            pad_token_id=128001,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=15,
            repetition_penalty=1.1,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p 
        )[0]
        output_ids = output_ids[len(data["input_ids"][0]) :]
        output = tokenizer.decode(output_ids, skip_special_tokens=True)
        return output.strip()
    
    response = generate(model, tokenizer, prompt)
    
    return response