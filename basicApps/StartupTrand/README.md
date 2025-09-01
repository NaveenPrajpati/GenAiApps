# create and activate a virtual environment (recommended)

python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate

# install dependencies

pip install -U streamlit langchain langchain-anthropic langchain-community \
 langchain-text-splitters duckduckgo-search newspaper3k unstructured tiktoken

# (If parsing HTML is flaky in your env, consider:)

# pip install "unstructured[all]"
