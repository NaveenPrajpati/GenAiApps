## ​ → 🎙️ Blog to Podcast App

A modular Streamlit app that lets users convert public blog posts into podcast-style audio. The “Blog to Podcast” module scrapes blog content, summarizes it using OpenAI GPT-4 (or any other preferred llm), and offers optional audio generation using a TTS provider like Murf (or any other preferred service).

### Features

- **Modular Architecture**: Easily extendable—each feature (Blog to Podcast, About, Contact) is encapsulated as a separate agent inside `basicApps/BlogToPodcast/app.py`.

- **Blog Scraping**: Fetches and parses public blog content directly from the provided URL using LangChain’s `WebBaseLoader`.

- **Summary Generation**: Produces a concise, engaging summary (configurable length) via OpenAI’s GPT-4 model.

- **Optional Audio (TTS)**: Converts the generated summary into speech using Murf’s TTS SDK (optional)—or switch to an alternative TTS service if preferred.

- **Sidebar Navigation**: Main app offers a clean UI to switch between "Blog to Podcast", "About", and "Contact" sections.

### Requirements

- Python 3.8+
- OpenAI API key
- (Optional) Murf API key (or alternative TTS provider credentials)
- Environment management tools like `python-dotenv` or Streamlit secrets

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/your-repo
   cd your-repo
   ```

2. Install the required Python packages:
   ```bash
   pip install streamlit langchain langchain-openai langchain-community python-dotenv murf
   ```
3. Create a .env file in your project root:
   ```bash
   OPENAI_API_KEY=your_openai_key_here
   MURF_API_KEY=your_murf_key_here  # only if using Murf for TTS
   ```
4. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
