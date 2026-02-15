# LangChain RAG Agent with Hybrid Retrieval

A LangChain-based conversational agent with Retrieval-Augmented Generation (RAG), hybrid retrieval, tool use, and long-term memory retention.

## Features

- **Hybrid Retrieval**: Combines semantic search (vector similarity) and keyword search for better results
- **Multiple Tools**: Flight Schedule, Hotel Search, Currency Converter, and RAG Query tools
- **Long-term Memory**: Conversation history stored in vector database for future retrieval
- **Document Indexing**: Automatically loads and indexes files from `data/` directory
- **OpenRouter LLM**: Uses NVIDIA Nemotron model via OpenRouter
- **HuggingFace Embeddings**: sentence-transformers/all-MiniLM-L6-v2 for text embeddings
- **ChromaDB Vector Store**: Persistent vector storage for documents and conversations

## Requirements

- Python 3.8+
- OpenRouter API key
- HuggingFace API key (optional, for model downloads)

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd langchain-rag-agent
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` file from example:

```bash
cp .env-example .env
```

4. Edit `.env` and add your API keys:

```
OPENROUTER_API_KEY=your_actual_openrouter_key
HF_API_KEY=your_huggingface_key
```

## Usage

### Basic Usage

Run the agent with a query:

```bash
python main.py "Get flight from Lagos to Nairobi"
```

### Examples

1. **Flight Schedule**:

```bash
python main.py "Get flight from Lagos to Nairobi"
```

2. **Hotel Search**:

```bash
python main.py "Show me hotels in Nairobi"
```

3. **Currency Conversion**:

```bash
python main.py "Convert 920 USD to NGN"
```

4. **RAG Query** (queries indexed documents):

```bash
python main.py "What information do you have about our company?"
```

### Adding Documents to RAG

Place any text files in the `data/` directory. The agent will automatically:

1. Load all files on startup
2. Split them into chunks
3. Index them in the vector store
4. Make them searchable via the RAG_Query tool

Example:

```bash
echo "Our company was founded in 2020." > data/company_info.txt
python main.py "When was our company founded?"
```

## Architecture

### Components

1. **OpenRouterLLM**: Custom LLM wrapper for OpenRouter API
2. **HybridRetriever**: Combines semantic and keyword search
3. **RAGAgent**: Main agent orchestrator
4. **Tools**:
    - get_flight_schedule: Returns flight duration and price
    - get_hotel_schedule: Returns hotel options for a city
    - convert_currency: Converts between currencies
    - RAG_Query: Queries vector store for information

### Data Flow

```
User Input → Agent → Tool Selection → Tool Execution → Response
                ↓
         Conversation History
                ↓
         Vector Store (Long-term Memory)
```

### Conversation History

- **In-Memory**: Stored in list for current session display
- **Vector Store**: Persisted for long-term retrieval across sessions
- Each conversation turn is indexed and searchable

## Directory Structure

```
langchain-rag-agent/
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── .env-example        # Example environment variables
├── .env                # Your actual environment variables (not committed)
├── README.md           # This file
├── data/               # Place your documents here
│   └── (your .txt files)
└── chroma_db/          # ChromaDB persistence directory (auto-created)
    └── (database files)
```

## Configuration

Environment variables in `.env`:

| Variable           | Required | Default     | Description             |
| ------------------ | -------- | ----------- | ----------------------- |
| OPENROUTER_API_KEY | Yes      | -           | Your OpenRouter API key |
| HF_API_KEY         | No       | -           | HuggingFace API key     |
| CHROMA_PERSIST_DIR | No       | ./chroma_db | Vector store directory  |
| DATA_DIR           | No       | ./data      | Documents directory     |

## Output Format

The script prints to stdout:

1. **Conversation History**: All previous turns
2. **Current Query**: The user's input
3. **Agent Processing**: Tool calls and reasoning (verbose mode)
4. **Final Response**: The agent's answer

Example output:

```
================================================================================
CONVERSATION HISTORY:
================================================================================
(No previous conversation)

================================================================================
CURRENT QUERY:
================================================================================
User: What is the weather in Jos?

================================================================================
AGENT PROCESSING:
================================================================================
[Agent reasoning and tool calls...]

================================================================================
FINAL RESPONSE:
================================================================================
Assistant: Jos, Nigeria: 24°C, Partly cloudy with a chance of afternoon showers
================================================================================
```

## Advanced Usage

### Custom Tools

Add your own tools by modifying the `create_tools()` method in `main.py`:

```python
def custom_tool(self, input: str) -> str:
    """Your custom tool logic"""
    return "Tool result"

# Add to create_tools():
Tool(
    name="CustomTool",
    func=self.custom_tool,
    description="Description of what the tool does"
)
```

### Adjusting Retrieval

Modify retrieval parameters in `HybridRetriever`:

```python
self.retriever = HybridRetriever(
    vectorstore=self.vectorstore,
    k=10  # Increase number of retrieved documents
)
```

## Troubleshooting

### Issue: "OPENROUTER_API_KEY not found"

**Solution**: Ensure `.env` file exists and contains valid API key

### Issue: No documents loaded

**Solution**: Check that files exist in `data/` directory and are readable

### Issue: ChromaDB errors

**Solution**: Delete `chroma_db/` directory and restart

### Issue: Slow performance

**Solution**:

- Reduce number of documents in `data/`
- Decrease chunk size in text splitter
- Use fewer retrieval results (lower `k` value)

## Contributing

Pull requests welcome! Please ensure:

1. Code follows PEP 8 style guide
2. All tests pass
3. Documentation is updated

## Support

For issues and questions, please open a GitHub issue.
