import os
import csv
import socket
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pipeline import build_rag_pipeline, handle_query  # Import your functions

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the RAG pipeline
vector_store = build_rag_pipeline()

def find_free_port():
    """Finds a free port dynamically."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))  # Bind to any available port
        return s.getsockname()[1]  # Return assigned port

@app.route('/')
def home():
    """Serve the frontend index.html file."""
    return send_from_directory('frontend', 'index.html')

@app.route('/query', methods=['POST'])
def query():
    """Handles user queries and returns the AI-generated response."""
    try:
        data = request.json
        query_text = data.get('query')

        if not query_text:
            return jsonify({'error': 'Query not provided'}), 400

        response = handle_query(query_text, vector_store)
        return jsonify({'response': response})
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/contact', methods=['POST'])
def contact():
    """Handles contact form submissions and saves them to a CSV file."""
    try:
        data = request.json
        name = data.get('name')
        email = data.get('email')
        message = data.get('message')

        if not name or not email or not message:
            return jsonify({'error': 'All fields are required!'}), 400

        os.makedirs('data', exist_ok=True)  # Ensure the data directory exists
        csv_file_path = 'data/contact_responses.csv'

        if not os.path.exists(csv_file_path):
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['name', 'email', 'message'])  # Write headers

        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([name, email, message])

        return jsonify({'message': 'Contact information received and saved!'}), 200

    except Exception as e:
        return jsonify({'error': f'Failed to save contact information: {str(e)}'}), 500

if __name__ == '__main__':
    PORT = find_free_port()
    print(f"🚀 Running Flask on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
