from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Set up the environment variable for HuggingFace and initialize the desired model.
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

conversation_model_name =  "meta-llama/Llama-3.2-3B-Instruct" 
conversation_tokenizer = AutoTokenizer.from_pretrained(conversation_model_name, token=HF_TOKEN)
conversation_model = AutoModelForCausalLM.from_pretrained(conversation_model_name, token=HF_TOKEN)

def get_conversation_response(prompt, history):
    if history==None:
        prompt = (
            f"### Instructions ###\n"
            f"You are a helpful assistant called The Smarty One (unless otherwise stated by the user). Answer the user's question or respond to their statement: {prompt}.\n"
            f"Start your response with 'Assistant: ' without restating the instructions.\n"
        )
    
    # Encode the input with an attention mask
    input_ids = conversation_tokenizer.encode(prompt + conversation_tokenizer.eos_token, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)

    # Combine with history if available
    if history is not None:
        bot_input_ids = torch.cat([history, input_ids], dim=-1)
        attention_mask = torch.cat([torch.ones_like(history), attention_mask], dim=-1)
    else:
        bot_input_ids = input_ids

    # Generate the response
    response = conversation_model.generate(
        bot_input_ids, 
        attention_mask=attention_mask, 
        max_length=1000, 
        pad_token_id=conversation_tokenizer.eos_token_id
    )
    
    response = conversation_tokenizer.decode(response[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Decode and return the response and history
    return response.split("Assistant: ")[-1].strip(), bot_input_ids