# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, OPTForCausalLM
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import GPT2Tokenizer, AutoModelForCausalLM,  AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(model)

print(output.logits)
# print(encoded_input)

# word 50257 and the probabilty
# This basically takes all the items in the first dimension which is the batch and takes the last item in the list
distribution = output.logits[:, -1]
nextToken = torch.argmax(distribution, dim=1)
text = tokenizer.decode(nextToken)

print(output.logits[:, -1].shape)

next_token_tensor = torch.tensor([[nextToken]])
# print(encoded_input)
# print(encoded_input.input_ids.shape)
# Update Input Ids
input_ids = torch.clone(encoded_input["input_ids"])
input_ids = torch.cat((input_ids, next_token_tensor),  1)

# Attention Ids
new_attention_value = torch.tensor([[1]])
attention_mask = torch.clone(encoded_input["attention_mask"])
attention_mask = torch.cat((attention_mask, new_attention_value),  1)

print(input_ids)
print(attention_mask)
encoded_input2 = {"attention_mask": attention_mask, "input_ids": input_ids}
print(encoded_input)
print(encoded_input2)

output = model(**encoded_input2)
distribution = output.logits[:, -1]
softmax = torch.nn.Softmax()
# Softmax is used to normalize the outputs to a probability mass function
distribution = softmax(distribution)
print(distribution)
nextToken = torch.argmax(distribution, dim=1)
text = tokenizer.decode(nextToken)
print(nextToken)
print(text)

# Generate Tokens
# Input will be the max new tokens to generate
# Returns new Encoded Input


def generateToken(max_new_tokens: int, encoded_input: dict) -> dict:
    new_encoded_input = encoded_input
    for i in range(max_new_tokens):
        output = model(**new_encoded_input)
        distribution = output.logits[:, -1]
        nextToken = torch.argmax(distribution, dim=1)

        # Update Input Ids
        next_token_tensor = torch.tensor([[nextToken]])
        input_ids = torch.clone(new_encoded_input["input_ids"])
        input_ids = torch.cat((input_ids, next_token_tensor),  1)

        # Attention Ids
        new_attention_value = torch.tensor([[1]])
        attention_mask = torch.clone(new_encoded_input["attention_mask"])
        attention_mask = torch.cat((attention_mask, new_attention_value),  1)
        new_encoded_input = {
            "attention_mask": attention_mask, "input_ids": input_ids}
    return new_encoded_input


new_text = generateToken(10, encoded_input)
print(new_text)

text = tokenizer.batch_decode(new_text["input_ids"])
print(text)

# Importing necessary library
# Loading the pre-trained BERT tokenizer
# tokenizer_expert = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
# model_expert = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer_expert = GPT2Tokenizer.from_pretrained('gpt2')
model_expert = AutoModelForCausalLM.from_pretrained('gpt2')
# print(model)
# Input text
input_text = "Hello, how are you doing today?"

text = "Barack Obama was born in honolulu, hawaii. He was born in"
encoded_input = tokenizer(text, return_tensors='pt')
output = model_expert(**encoded_input)
print(output.logits)
logits = output.logits
distribution = output.logits[:, -1]
print(distribution.shape)

probabilities = torch.nn.functional.softmax(distribution, dim=-1)
ind = torch.argmax(probabilities, dim=1)


def generateTokenExpert(max_new_tokens: int, encoded_input: dict) -> dict:
    new_encoded_input = encoded_input
    for i in range(max_new_tokens):
        output = model_expert(**new_encoded_input)
        distribution = output.logits[:, -1]
        nextToken = torch.argmax(distribution, dim=1)

        # Update Input Ids
        next_token_tensor = torch.tensor([[nextToken]])
        input_ids = torch.clone(new_encoded_input["input_ids"])
        input_ids = torch.cat((input_ids, next_token_tensor),  1)

        # Attention Ids
        new_attention_value = torch.tensor([[1]])
        attention_mask = torch.clone(new_encoded_input["attention_mask"])
        attention_mask = torch.cat((attention_mask, new_attention_value),  1)
        new_encoded_input = {
            "attention_mask": attention_mask, "input_ids": input_ids}
    return new_encoded_input


text = "Barack Obama was born in honolulu, hawaii and moved to China. He was born in the city of"


model_am = OPTForCausalLM.from_pretrained("facebook/opt-125m")
tokenizer_am = AutoTokenizer.from_pretrained("facebook/opt-125m")

inputs = tokenizer_am(text, return_tensors="pt")
outputs = model_am(**inputs)
print(outputs.logits.shape)


def generateTokenAm(max_new_tokens: int, encoded_input: dict) -> dict:
    new_encoded_input = encoded_input
    for i in range(max_new_tokens):
        output = model_am(**new_encoded_input)
        distribution = output.logits[:, -1]
        nextToken = torch.argmax(distribution, dim=1)

        # Update Input Ids
        next_token_tensor = torch.tensor([[nextToken]])
        input_ids = torch.clone(new_encoded_input["input_ids"])
        input_ids = torch.cat((input_ids, next_token_tensor),  1)

        # Attention Ids
        new_attention_value = torch.tensor([[1]])
        attention_mask = torch.clone(new_encoded_input["attention_mask"])
        attention_mask = torch.cat((attention_mask, new_attention_value),  1)
        new_encoded_input = {
            "attention_mask": attention_mask, "input_ids": input_ids}
    return new_encoded_input


encoded_input = tokenizer(text, return_tensors='pt')

newTokens = generateToken(15, encoded_input)
print(tokenizer.batch_decode(newTokens["input_ids"]))

# print(tokenizer_expert.get_vocab())
# print(tokenizer_am.get_vocab())
# create an array where [2,0,1] represents the order in which I should grab from amateur
amateur = tokenizer_am.get_vocab()
expert_vocab = tokenizer_expert.get_vocab().items()
count = 0
reorder = {}
for key in tokenizer_expert.get_vocab().keys():
    if key in amateur:
        count += 1


def reArrangeAm(oldTensor):
    newOrder = [0] * len(tokenizer_expert.get_vocab().items())
    for key, value in tokenizer_expert.get_vocab().items():
        newOrder[value] = amateur[key]

    index_array = torch.tensor(newOrder)
    reorder_tensor = torch.index_select(oldTensor, 1, index_array)
    return reorder_tensor


def genExpert(new_encoded_input1):
    output = model_expert(**new_encoded_input1)
    distribution = output.logits[:, -1]
    # Need to recalculate the encoded input each time based on what is outputed
    return distribution


def genAmateur(new_encoded_input1):
    output = model_am(**new_encoded_input1)
    distribution = output.logits[:, -1]
    # Need to recalculate the encoded input each time based on what is outputed
    return distribution


def gen(encoded_input, encoded_input2):
    # Encode Am
    # Encode Expert
    exp_dist = genExpert(encoded_input)
    am_dist = genAmateur(encoded_input2)
    probs_amateur_reordered = reArrangeAm(am_dist)
    log_probs_amateur_reordered = torch.nn.functional.log_softmax(
        probs_amateur_reordered, dim=1)
    log_probs_expert = torch.nn.functional.log_softmax(exp_dist, dim=1)
    probs_expert = torch.nn.functional.softmax(exp_dist, dim=1)
    assert log_probs_amateur_reordered.shape == log_probs_expert.shape
    threshold = torch.max(probs_expert) * 0.1
    mask = probs_expert >= threshold
    print(mask)
    result = torch.full_like(log_probs_expert, float('-inf'))
    # Perform subtraction where mask is True
    result[mask] = log_probs_expert[mask] - log_probs_amateur_reordered[mask]
    nextToken = torch.argmax(result)
    return nextToken

# gen()


tokens = 15
text = "Barack Obama was born in honolulu, hawaii. He was born in"
encoded_input = tokenizer_expert(text, return_tensors='pt')
encoded_input2 = tokenizer_am(text, return_tensors='pt')
for i in range(tokens):
    next_token = gen(encoded_input, encoded_input2)
    print("TokeN", next_token)
    # encoded_input expert
    next_token_tensor = torch.tensor([[next_token]])
    input_ids = torch.clone(encoded_input['input_ids'])
    input_ids = torch.cat((input_ids, next_token_tensor),  1)
    encoded_input = {"input_ids": input_ids}
    # Encoded Am
    token_decoded = tokenizer_expert.decode(next_token)
    next_token_tensor_am = tokenizer_am(token_decoded, return_tensors='pt')
    # print(next_token_tensor_am)
    # print(token_decoded)
    input_ids_am = torch.clone(encoded_input2['input_ids'])
    input_ids_am = torch.cat(
        (input_ids_am, next_token_tensor_am['input_ids']),  1)
    encoded_input2 = {"input_ids": input_ids_am}

print(tokenizer_expert.batch_decode(encoded_input['input_ids']))

encoded_input = tokenizer_expert(text, return_tensors='pt')

newTokens_expert = generateTokenExpert(15, encoded_input)
print(tokenizer_expert.batch_decode(newTokens_expert["input_ids"]))

encoded_input = tokenizer_am(text, return_tensors='pt')

newTokens = generateTokenAm(15, encoded_input)
print(tokenizer_am.batch_decode(newTokens["input_ids"]))
