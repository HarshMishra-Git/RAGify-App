# import openai

# openai.api_key = "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"

# def generate_response(query, context, model="gpt-3.5-turbo"):
#     messages = [{"role": "system", "content": "Use the provided context to answer the query."},
#                 {"role": "user", "content": f"Context: {context}\nQuery: {query}"}]
#     response = openai.ChatCompletion.create(model=model, messages=messages)
#     return response["choices"][0]["message"]["content"]
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # Load the LLaMA model and tokenizer from Hugging Face
# MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Change to a smaller model if needed
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# def generate_response(query, context):
#     """Generates a response based on the query and retrieved context."""
#     prompt = f"Context: {context}\n\nUser: {query}\n\nAI:"
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Move to GPU if available
#     output = model.generate(**inputs, max_length=512, temperature=0.7, top_p=0.9)
#     response = tokenizer.decode(output[0], skip_special_tokens=True)
    
#     return response

from transformers import pipeline

# Initialize Hugging Face pipeline for text generation
model_name = "gpt2"  # You can replace this with another model like 'gpt-neo', 'distilgpt2', etc.
generator = pipeline('text-generation', model=model_name)

def generate_response(query, context):
    """
    Generates a response based on the provided query and context using Hugging Face's model.
    
    Parameters:
    - query (str): The user's query.
    - context (str): The context to be considered for the response generation.

    Returns:
    - str: The generated response.
    """
    input_text = context + " " + query  # Combine the query and context
    response = generator(input_text, max_length=150, num_return_sequences=1)
    return response[0]['generated_text']

# Example usage (replace this with your actual query and context)
if __name__ == "__main__":
    context = "Artificial Intelligence (AI) is a field of computer science that aims to create machines that can perform tasks that typically require human intelligence."
    query = "What is AI?"
    
    answer = generate_response(query, context)
    print(f"Answer: {answer}")
