from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
import os
import PyPDF2
from transformers import pipeline

app = Flask(__name__)

# Set a secret key for session management
app.config['SECRET_KEY'] = 'cenas_key'

# Set upload folder and allowed file types
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load HuggingFace pipelines
summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

# Summarize the document with error handling
def summarize_text(text, max_length=200, min_length=50):
    # Ensure the input text is non-empty
    if not text.strip():
        return "The document is empty or contains unreadable content."
    
    # HuggingFace models usually accept up to 1024 tokens; truncate if necessary
    token_limit = 1024  # Adjust based on your model's limit
    truncated_text = text[:token_limit]  # Truncate the text

    try:
        # Use the summarization model
        summary_result = summarizer(truncated_text, max_length=max_length, min_length=min_length, do_sample=False)

        # Check if the summarizer returned results
        if summary_result and len(summary_result) > 0:
            return summary_result[0]['summary_text']
        else:
            return "The summarization model did not return any results."
    except Exception as e:
        # Handle unexpected errors
        return f"An error occurred during summarization: {str(e)}"

# Generate a response based on the document text and user query
def generate_response(query, document_text):
    # Implement simple keyword matching or more sophisticated query handling based on the document text
    if query.lower() in document_text.lower():
        return "I found relevant information in the document. Let me elaborate..."
    else:
        return "Sorry, I couldn't find relevant information in the document. Could you rephrase your question?"

# Helper function to check file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to render the chat interface
@app.route('/')
def index():
    return render_template('index_1.html')

# Route for file upload and summarization
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Extract text from PDF
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ''.join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

        # Store the document text in the session
        session['document_text'] = text

        # Summarize the document
        summary = summarize_text(text)

        return jsonify({'summary': summary}), 200

    return jsonify({'error': 'Invalid file type'}), 400

# Route for chat
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    document_text = session.get('document_text', '')

    if not document_text:
        return jsonify({'response': 'No document uploaded. Please upload a document first.'}), 400

    print('user: ', user_message)
    print('file: ', document_text)
    
    # Generate a response using the stored text and user query
    response = generate_response(user_message, document_text)
    return jsonify({'response': response}), 200


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=False)
