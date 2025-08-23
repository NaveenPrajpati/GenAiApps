import os
import base64
import tempfile
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()


def encode_image_to_base64(image_path):
    """Convert image to base64 string for API transmission"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def medicalImageDiagnose():
    """Main function for medical image diagnosis"""

    st.title("🏥 Medical Imaging Diagnosis Assistant")
    st.write("Upload a medical image for AI-powered analysis")

    # Add disclaimer
    st.warning(
        "⚠️ **Important Disclaimer**: This tool provides AI-generated analysis for educational purposes only. Always consult qualified healthcare professionals for medical diagnosis and treatment."
    )
    # Initialize tools
    search = DuckDuckGoSearchRun()

    # Medical Analysis Query Template
    query_template = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. 
Analyze the provided medical image and structure your response as follows:

### 1. Image Type & Region
- Specify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)
- Identify the anatomical region and positioning
- Comment on image quality and technical adequacy

### 2. Key Findings
- List primary observations systematically
- Note any abnormalities with precise descriptions
- Include measurements and densities where relevant
- Describe location, size, shape, and characteristics
- Rate severity: Normal/Mild/Moderate/Severe

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level
- List differential diagnoses in order of likelihood
- Support each diagnosis with observed evidence
- Note any critical or urgent findings

### 4. Patient-Friendly Explanation
- Explain findings in simple, clear language
- Avoid medical jargon or provide clear definitions
- Include visual analogies if helpful
- Address common patient concerns

### 5. Research Context
Use available search tools to find:
- Recent medical literature about similar cases
- Standard treatment protocols
- Relevant medical references
- Current best practices

IMPORTANT DISCLAIMER: This analysis is for educational purposes only and should not replace professional medical consultation.

Format your response using clear markdown headers and bullet points.
"""

    # Create containers for better organization
    upload_container = st.container()
    image_container = st.container()
    analysis_container = st.container()

    with upload_container:
        uploaded_file = st.file_uploader(
            "Upload Medical Image",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
            help="Supported formats: JPG, JPEG, PNG, BMP, TIFF",
        )

    if uploaded_file is not None:
        with image_container:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                try:
                    # Open and process the image
                    image = Image.open(uploaded_file)

                    # Convert to RGB if necessary
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    # Resize image for display
                    width, height = image.size
                    aspect_ratio = width / height
                    new_width = 500
                    new_height = int(new_width / aspect_ratio)
                    resized_image = image.resize(
                        (new_width, new_height), Image.Resampling.LANCZOS
                    )

                    st.image(
                        resized_image,
                        caption="Uploaded Medical Image",
                        use_container_width=True,
                    )

                    analyze_button = st.button(
                        "🔍 Analyze Image", type="primary", use_container_width=True
                    )

                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    return

        with analysis_container:
            if analyze_button:
                with st.spinner("🔄 Analyzing image... Please wait."):
                    try:
                        # Check if OpenAI API key is available
                        if not os.getenv("OPENAI_API_KEY"):
                            st.error(
                                "OpenAI API key not found. Please set your OPENAI_API_KEY environment variable."
                            )
                            return

                        # Save image temporarily
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".png"
                        ) as temp_file:
                            resized_image.save(temp_file.name, format="PNG")
                            temp_path = temp_file.name

                        # Encode image to base64
                        base64_image = encode_image_to_base64(temp_path)

                        # Initialize LLM with vision capabilities
                        llm = ChatOpenAI(
                            model="gpt-4o",  # Use GPT-4 Vision
                            temperature=0.1,
                            max_tokens=2000,
                        )

                        # Create the message with image
                        message = HumanMessage(
                            content=[
                                {"type": "text", "text": query_template},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    },
                                },
                            ]
                        )

                        # Get analysis
                        response = llm.invoke([message])

                        # Display results
                        st.markdown("### 📋 Analysis Results")
                        st.markdown("---")
                        st.markdown(response.content)

                        # Additional research section
                        # st.markdown("### 🔍 Additional Research")
                        # with st.expander(
                        #     "Click to search for related medical literature"
                        # ):
                        #     search_query = st.text_input(
                        #         "Enter search terms for medical literature:",
                        #         placeholder="e.g., chest X-ray pneumonia diagnosis",
                        #     )
                        #     if st.button("Search") and search_query:
                        #         with st.spinner("Searching..."):
                        #             try:
                        #                 search_results = search.invoke(search_query)
                        #                 st.markdown("**Search Results:**")
                        #                 st.write(search_results)
                        #             except Exception as e:
                        #                 st.error(f"Search error: {str(e)}")

                        st.markdown("---")
                        st.caption(
                            "⚕️ **Medical Disclaimer**: This AI analysis is for educational and research purposes only. "
                            "It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. "
                            "Always seek the advice of qualified healthcare professionals for any medical concerns."
                        )

                        # Clean up temporary file
                        try:
                            os.unlink(temp_path)
                        except:
                            pass

                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")
                        st.info(
                            "Please check your OpenAI API configuration and try again."
                        )
    else:
        st.info("👆 Please upload a medical image to begin analysis")

        # Add helpful information
        with st.expander("ℹ️ How to use this tool"):
            st.markdown(
                """
            **Steps to analyze medical images:**
            1. Upload a medical image (X-ray, MRI, CT scan, etc.)
            2. Click 'Analyze Image' to get AI-powered analysis
            3. Review the structured diagnostic assessment
            4. Use the research section for additional medical literature
            
            **Supported formats:** JPG, JPEG, PNG, BMP, TIFF
            
            **Remember:** This tool is for educational purposes only!
            """
            )
