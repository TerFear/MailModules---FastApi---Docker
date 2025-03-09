import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from fastapi import FastAPI

app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"{device}: определил девайс")

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"{torch_dtype}: определи ещё девайс")
MODEL_NAME = "IlyaGusev/saiga_llama3_8b"

DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
print("начал загрузку модели")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    device_map=device
)

print("Закончил загрузку моедели")
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)




@app.get("/summarization/{question}")
def processing(question):


    inputs = [f"{question}"]
    print(f"Вопрос был успешно получен:{question}")

    for query in inputs:
        prompt = tokenizer.apply_chat_template([{
            "role": "system",
            "content": DEFAULT_SYSTEM_PROMPT
        }, {
            "role": "user",
            "content": query
        }], tokenize=False, add_generation_prompt=True)
        data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        data = {k: v.to(model.device) for k, v in data.items()}
        output_ids = model.generate(**data, generation_config=generation_config)[0]
        output_ids = output_ids[len(data["input_ids"][0]):]
        output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        print(f"Ответ на поставленный вопрос: {output}")

        return {"result":output}
