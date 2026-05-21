from betterairllm import BetterAirLLMLlama2

MAX_LENGTH = 128

model = BetterAirLLMLlama2("garage-bAInd/Platypus2-70B-instruct")


input_text = [
        'What is the capital of United States?',

    ]

input_tokens = model.tokenizer(input_text,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=MAX_LENGTH,
    padding=True)

generation_output = model.generate(
    input_tokens['input_ids'].cuda(),
    max_new_tokens=2,
    use_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])

print(output)
