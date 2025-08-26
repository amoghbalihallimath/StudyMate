# PDF Q&A Assistant with Gemini

An interactive web application that allows users to upload PDF documents, ask questions about their content, generate summaries, and create quizzes based on the document content using Google's Gemini API and Hugging Face models.

## Features

- 📄 Upload and process PDF documents
- 💬 Chat interface for asking questions about the document
- 📝 Generate concise summaries of the document
- ❓ Create interactive quizzes based on the document content
- 📂 Track and switch between multiple uploaded documents
- 🔍 Semantic search for relevant context within documents

## Prerequisites

- Python 3.8+
- Google Gemini API key
- (Optional) Hugging Face API token (for private models)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd pdf-qa-assistant
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your API keys:
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload a PDF document using the sidebar

4. Use the different tabs to:
   - Chat: Ask questions about the document
   - Summarize: Generate a summary of the document
   - Quiz: Create a quiz based on the document content

## How It Works

1. **Document Processing**:
   - PDFs are processed to extract text
   - Text is split into manageable chunks
   - Each chunk is converted to embeddings using Sentence Transformers
   - A FAISS index is created for efficient similarity search

2. **Question Answering**:
   - User questions are matched with relevant document sections
   - Context is sent to Gemini API for generating accurate answers

3. **Quiz Generation**:
   - Document content is analyzed to create relevant questions
   - Multiple-choice questions are generated with correct answers

## Dependencies

- Streamlit - Web application framework
- PyPDF2 - PDF text extraction
- Google Generative AI - Access to Gemini models
- Sentence Transformers - Text embeddings
- FAISS - Efficient similarity search
- python-dotenv - Environment variable management

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Gemini for the powerful language model
- Hugging Face for the open-source models and tools
- Streamlit for the easy-to-use web framework
