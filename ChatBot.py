from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load the GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Define a function to generate a response to a user input
def generate_response(input_text):
    # Encode the input text into a tensor
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate a response
    response = model.generate(input_ids, max_length=100, top_p=0.9, top_k=40)

    # Decode the generated response
    return tokenizer.decode(response[0], skip_special_tokens=True)

# Test the chatbot
response = generate_response("What is the weather like today?")
print(response)
