from pprint import pprint
from argparse import ArgumentParser

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM


transformers.logging.set_verbosity_error()

# generation config for Llama 3
generation_config = {
    "max_length": 128,
    "early_stopping": False,
    "num_return_sequences": 1,
    "do_sample": False,
    "top_p": 1.0,
    "temperature": None,
    "bos_token_id": 128000,
    "pad_token_id": 128001,
    "eos_token_id": 128001,
    "decoder_start_token_id": 128000,
    "use_cache": True,
}

model = AutoModelForCausalLM.from_pretrained("/data2/checkpoints/moe/LM-Lexicon-4xLlama-3-8b")

tokenizer = AutoTokenizer.from_pretrained("/data2/checkpoints/moe/LM-Lexicon-4xLlama-3-8b")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--prompt", type=str, default=None)

    args = parser.parse_args()

    if args.prompt is None:
        args.prompt = "<|begin_of_text|>\"prime a cannon\" What is the definition of \"prime\"?\n\n"

    inputs = tokenizer(args.prompt, return_tensors="pt")

    outputs = model.generate(**inputs, **generation_config)

    decoded_outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    for output in decoded_outputs:
        print(output)
