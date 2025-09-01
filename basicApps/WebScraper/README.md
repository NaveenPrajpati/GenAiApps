# LangChain Web Scraper

**A simple Streamlit app that lets you input a website URL and an extraction prompt—then uses LangChain to scrape and extract structured information from the page.**

## Features

- **User-friendly UI:** Powered by Streamlit; input a URL and describe what you want to extract.
- **Web page loading:** Uses `WebBaseLoader` to fetch content from the website.
- **Chunking for long pages:** Splits content using `RecursiveCharacterTextSplitter` to handle long documents.
- **Structured extraction with LLMs:** Uses OpenAI (or other providers) with `with_structured_output()` and a Pydantic model (`ExtractionResult`) to parse output into clean, validated data.  
  This aligns with current best practices in LangChain—where structured output enables robust schema-based extraction. [oai_citation:0‡LangChain](https://python.langchain.com/docs/concepts/structured_outputs/?utm_source=chatgpt.com) [oai_citation:1‡Medium](https://medium.com/%40obaff/python-ai-web-scraper-tutorial-35e0cc6f7398?utm_source=chatgpt.com) [oai_citation:2‡GitHub](https://github.com/langchain-ai/langchain/issues/32687?utm_source=chatgpt.com)
- **Clear, organized results:** Presents summaries, extracted items, and evidence snippets in the UI.

---

## How It Works

1. **Define a schema**  
   A `Pydantic` class, `ExtractionResult`, defines the expected output structure.

2. **Load and chunk content**  
   `WebBaseLoader` grabs the page content, and `RecursiveCharacterTextSplitter` breaks it into manageable chunks.

3. **Use the LLM with structured output**  
   With OpenAI (or another supported provider), we do:
   ```python
   llm = ChatOpenAI(...).with_structured_output(ExtractionResult)
   ```
