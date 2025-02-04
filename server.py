from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # Import CORS for handling cross-origin requests
from pipeline import build_rag_pipeline, handle_query  # Import your functions
import csv
import os

app = Flask(__name__)  # Initialize Flask app
# CORS(app, resources={r"/*": {"origins": "http://localhost"}})  # Enable CORS for frontend on localhost
CORS(app)
# Initialize the RAG pipeline
vector_store = build_rag_pipeline()

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

        # Ensure the data directory exists
        os.makedirs('data', exist_ok=True)
        csv_file_path = 'data/contact_responses.csv'

        # Ensure the CSV file exists with headers
        if not os.path.exists(csv_file_path):
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['name', 'email', 'message'])  # Write headers

        # Write contact data to CSV
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([name, email, message])

        return jsonify({'message': 'Contact information received and saved!'}), 200

    except Exception as e:
        return jsonify({'error': f'Failed to save contact information: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app in debug mode
