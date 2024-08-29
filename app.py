from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize Flask app
app = Flask(__name__)

# Load the pretrained chatbot model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Define the chat route
@app.route('/chat', methods=['POST'])
def chat():
    # Get the user input from the POST request
    user_input = request.json['message']

    # Tokenize and encode the input
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # Generate the chatbot's response
    bot_output = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the output and return the response
    response = tokenizer.decode(bot_output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return jsonify({"response": response})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

