import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import gradio as gr
import nltk

# Download NLTK data
nltk.download('punkt')

# Load model and tokenizer
model_path = "./chatbot_model"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Text processing functions
def normalize_input(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\?\.,!]', '', text)
    return text

def capitalize_response(response):
    sentences = [s.capitalize() for s in response.split(". ") if s]
    return ". ".join(sentences)

# Main response function
def generate_response(query):
    try:
        normalized_query = normalize_input(query)
        input_text = f"generate response: {normalized_query}"
        input_ids = tokenizer(input_text, return_tensors="pt", 
                            truncation=True, max_length=128).input_ids.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=190,
                repetition_penalty=1.5,
                num_beams=3
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return capitalize_response(response)
    
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Simple Gradio Interface
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(
        label="Enter your query",
        placeholder="e.g. How to apply for a visa?",
        lines=3
    ),
    outputs=gr.Textbox(
        label="Response",
        lines=5,
        interactive=False
    ),
    title="Travel Guide Chatbot",
    description="This is a chatbot that helps with travel-related queries. Ask anything!",
    theme=gr.themes.Soft(),
    examples=[
        ["How can I find the cheapest flight"]
    ]
)

# Launch
iface.launch()