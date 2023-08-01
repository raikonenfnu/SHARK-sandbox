import torch
from transformers import AutoTokenizer, OPTForCausalLM

model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
logits = model.forward(inputs.input_ids).logits
# Logits are results before we put into softmax to get a probability distribution.
# TODO: Apply torch softmax to get probability distribution.
# TODO: Implement method to extract the token with highest probability.
# TODO: You can use:  tokenizer.batch_decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].
#       to get word out of token_id which is a number/int.
