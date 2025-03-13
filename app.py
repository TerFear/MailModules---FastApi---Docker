import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoModelForSpeechSeq2Seq, \
    AutoProcessor, pipeline
from fastapi import FastAPI, UploadFile,File

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




model_id = "openai/whisper-large-v3-turbo"

speech_model= AutoModelForSpeechSeq2Seq.from_pretrained( model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)

speech_model.to(device)
processor = AutoProcessor.from_pretrained(model_id)


pipe = pipeline(
        "automatic-speech-recognition",
        model=speech_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
)

print(f"Пайп лайн для нейросети 2 был успешно установлен")






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

@app.post("/trans/{file}")
def translate(file: UploadFile = File(...)):

    result = pipe(file)

    return {"result", result['text']}


