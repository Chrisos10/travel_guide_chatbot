import re
import torch
import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model and tokenizer (Ensure the 'travel_chatbot_model' directory is present in your repo)
model_path = "./chatbot_model"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Input normalization
def normalize_input(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\?\.,!]', '', text)
    return text

# Output formatting
def capitalize_response(response):
    sentences = response.split(". ")
    unique_sentences = []
    for s in sentences:
        if s and s not in unique_sentences:
            unique_sentences.append(s.capitalize())
    return ". ".join(unique_sentences)

# Inference function
def test_query(query):
    query_lower = normalize_input(query)
    input_text = f"generate response: Current query: {query_lower}"
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).input_ids.to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=190,
            temperature=0.8,
            top_k=70,
            repetition_penalty=1.5
        )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return capitalize_response(response)

# Gradio UI
iface = gr.Interface(
    fn=test_query,
    inputs=gr.Textbox(label="Enter your query", placeholder="Ask something...", lines=2),
    outputs=gr.Textbox(label="Response"),
    title="Travel Guide Chatbot",
    description="This is a chatbot that helps with travel-related queries. Ask anything!",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8080)
