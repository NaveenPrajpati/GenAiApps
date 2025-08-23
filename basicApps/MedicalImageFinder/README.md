# 🏥 Medical Imaging Diagnosis Assistant

A sophisticated AI-powered medical imaging analysis tool built with Streamlit, OpenAI GPT-4 Vision, and LangChain. This application provides comprehensive analysis of medical images including X-rays, MRI scans, CT scans, and ultrasounds with structured diagnostic assessments.

## ⚠️ **IMPORTANT MEDICAL DISCLAIMER**

**This application is for educational and research purposes only. It is NOT intended for clinical use and should NEVER replace professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical concerns.**

## 🌟 Features

- **AI-Powered Analysis**: Uses OpenAI's GPT-4 Vision model for comprehensive medical image analysis
- **Multiple Image Formats**: Supports JPG, JPEG, PNG, BMP, and TIFF formats
- **Structured Reporting**: Provides organized diagnostic assessments with:
  - Image type and region identification
  - Key findings and abnormalities
  - Diagnostic assessment with confidence levels
  - Patient-friendly explanations
  - Research context and literature references
- **Interactive Research**: Built-in DuckDuckGo search for medical literature
- **User-Friendly Interface**: Clean, responsive Streamlit interface
- **Image Processing**: Automatic image optimization and format conversion

### Installation

1. **Clone the repository:**

   ```bash
   git clone <your-repository-url>
   cd medical-imaging-diagnosis
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Run the application:**

   ```bash
   streamlit run app.py
   ```

6. **Access the application:**
   Open your web browser and navigate to `http://localhost:8501`

## 📦 Dependencies

Create a `requirements.txt` file with the following dependencies:

```txt
streamlit>=1.28.0
langchain>=0.1.0
langchain-openai>=0.1.0
langchain-community>=0.0.20
python-dotenv>=1.0.0
Pillow>=10.0.0
openai>=1.0.0
duckduckgo-search>=3.9.0
```

## 🛠️ Configuration

### Environment Variables

Create a `.env` file in your project root:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
OPENAI_MODEL=gpt-4o  # Default model for analysis
MAX_TOKENS=2000      # Maximum tokens for responses
TEMPERATURE=0.1      # Model temperature (0.0-1.0)
```

### OpenAI API Setup

1. Sign up at [OpenAI Platform](https://platform.openai.com/)
2. Create an API key in your dashboard
3. Add credits to your account (GPT-4 Vision requires paid usage)
4. Copy your API key to the `.env` file

## 📖 Usage Guide

### Basic Usage

1. **Launch the application:**

   ```bash
   streamlit run app.py
   ```

2. **Upload an image:**

   - Click "Browse files" or drag and drop your medical image
   - Supported formats: JPG, JPEG, PNG, BMP, TIFF

3. **Analyze the image:**

   - Click "🔍 Analyze Image" button
   - Wait for the AI analysis to complete

4. **Review results:**
   - Read the structured diagnostic assessment
   - Check the patient-friendly explanation
   - Use the research section for additional information

### Advanced Features

#### Research Integration

- Use the expandable research section to search for relevant medical literature
- Enter specific search terms related to your findings
- Get real-time search results from medical databases

#### Image Processing

- Images are automatically resized and optimized
- Format conversion happens automatically
- Original image quality is preserved for analysis

### Key Components

- **Streamlit Interface**: Web-based user interface
- **OpenAI GPT-4 Vision**: Core AI model for image analysis
- **LangChain**: Framework for AI application development
- **DuckDuckGo Search**: Research and literature search
- **PIL (Pillow)**: Image processing and optimization

## 🔧 API Reference

### Main Functions

#### `medical_image_diagnose()`

Main function that creates the Streamlit interface and handles user interactions.

#### `encode_image_to_base64(image_path)`

Converts image files to base64 encoding for API transmission.

- **Parameters**: `image_path` (str) - Path to the image file
- **Returns**: Base64 encoded string

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**⚕️ Remember: This tool is for educational purposes only. Always consult healthcare professionals for medical diagnosis and treatment decisions.**
