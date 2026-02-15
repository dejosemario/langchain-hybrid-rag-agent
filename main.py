#!/usr/bin/env python3
"""
LangChain Agent with RAG, Hybrid Retrieval, and Long-term Memory
Uses HuggingFace embeddings, OpenRouter LLM, and ChromaDB vector store
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

# For OpenRouter LLM
from langchain_community.llms import OpenAI
from langchain.llms.base import LLM
import requests

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DATA_DIR = os.getenv("DATA_DIR", "./data")

if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY not found in .env file")
    sys.exit(1)


class OpenRouterLLM(LLM):
    """Custom LLM wrapper for OpenRouter API"""
    
    model: str = "nvidia/nemotron-3-nano-30b-a3b:free"
    temperature: float = 0.7
    max_tokens: int = 2000
    
    @property
    def _llm_type(self) -> str:
        return "openrouter"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Call OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling OpenRouter API: {str(e)}"


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining semantic and keyword search"""
    
    vectorstore: Chroma
    k: int = 5
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get relevant documents using hybrid search"""
        # Semantic search using vector similarity
        semantic_docs = self.vectorstore.similarity_search(query, k=self.k)
        
        # Simple keyword search (filter by query terms in content)
        query_terms = query.lower().split()
        all_docs = self.vectorstore.get()
        
        keyword_docs = []
        if all_docs and 'documents' in all_docs:
            for idx, doc_text in enumerate(all_docs['documents']):
                if doc_text and any(term in doc_text.lower() for term in query_terms):
                    metadata = all_docs['metadatas'][idx] if 'metadatas' in all_docs else {}
                    keyword_docs.append(Document(page_content=doc_text, metadata=metadata))
        
        # Combine and deduplicate
        seen_content = set()
        combined_docs = []
        
        for doc in semantic_docs + keyword_docs[:self.k]:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                combined_docs.append(doc)
                if len(combined_docs) >= self.k:
                    break
        
        return combined_docs


class RAGAgent:
    """Main RAG Agent with tool-enhanced capabilities"""
    
    def __init__(self):
        self.setup_embeddings()
        self.setup_vectorstore()
        self.setup_llm()
        self.load_data_files()
        self.conversation_history = []
        
    def setup_embeddings(self):
        """Initialize HuggingFace embeddings"""
        print("Initializing embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
    def setup_vectorstore(self):
        """Initialize ChromaDB vector store"""
        print("Setting up vector store...")
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        
        self.vectorstore = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )
        
        # Setup hybrid retriever
        self.retriever = HybridRetriever(vectorstore=self.vectorstore, k=5)
        
    def setup_llm(self):
        """Initialize OpenRouter LLM"""
        print("Setting up LLM...")
        self.llm = OpenRouterLLM(
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            temperature=0.7,
            max_tokens=2000
        )
        
    def load_data_files(self):
        """Load all files from data/ directory into vector store"""
        print(f"Loading data files from {DATA_DIR}...")
        data_path = Path(DATA_DIR)
        
        if not data_path.exists():
            print(f"Creating data directory: {DATA_DIR}")
            data_path.mkdir(parents=True, exist_ok=True)
            return
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        
        documents = []
        for file_path in data_path.glob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                try:
                    print(f"Loading: {file_path.name}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Create document with metadata
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(file_path.name),
                            "type": "data_file",
                            "loaded_at": datetime.now().isoformat()
                        }
                    )
                    documents.append(doc)
                except Exception as e:
                    print(f"Error loading {file_path.name}: {e}")
        
        if documents:
            # Split documents into chunks
            split_docs = text_splitter.split_documents(documents)
            print(f"Adding {len(split_docs)} chunks to vector store...")
            self.vectorstore.add_documents(split_docs)
            self.vectorstore.persist()
            print(f"Successfully loaded {len(documents)} files")
        else:
            print("No data files found to load")
    
    def save_conversation_to_vectorstore(self, user_message: str, assistant_response: str):
        """Save conversation turn to vector store for long-term memory"""
        conversation_doc = Document(
            page_content=f"User: {user_message}\nAssistant: {assistant_response}",
            metadata={
                "type": "conversation",
                "timestamp": datetime.now().isoformat(),
                "user_message": user_message,
                "assistant_response": assistant_response
            }
        )
        self.vectorstore.add_documents([conversation_doc])
        self.vectorstore.persist()
    
    # Tool 1: Flight Schedule
    def get_flight_schedule(self, input_str: str) -> str:
        """Returns flight duration and price in USD"""
        try:
            # Parse input: expected format "origin to destination" or "origin, destination"
            parts = input_str.replace(" to ", ",").split(",")
            if len(parts) != 2:
                return "Error: Please provide origin and destination (e.g., 'Lagos to Nairobi' or 'Lagos, Nairobi')"
            
            origin = parts[0].strip()
            destination = parts[1].strip()
            
            # Simulated flight data
            flight_info = {
                "origin": origin,
                "destination": destination,
                "flight_time_hours": 5.5,
                "price_usd": 920
            }
            
            return f"Flight from {origin} to {destination}: {flight_info['flight_time_hours']} hours, ${flight_info['price_usd']} USD"
        except Exception as e:
            return f"Error getting flight schedule: {str(e)}"
    
    # Tool 2: Hotel Schedule
    def get_hotel_schedule(self, city: str) -> str:
        """Get hotel options for a city"""
        try:
            # Simulated hotel data
            hotels = {
                "nairobi": [
                    {"name": "Nairobi Serena", "price_usd": 250},
                    {"name": "Radisson Blu", "price_usd": 200}
                ],
                "lagos": [
                    {"name": "Eko Hotel", "price_usd": 180},
                    {"name": "Oriental Hotel", "price_usd": 150}
                ],
                "default": [
                    {"name": "City Hotel", "price_usd": 100},
                    {"name": "Budget Inn", "price_usd": 75}
                ]
            }
            
            city_key = city.lower().strip()
            hotel_list = hotels.get(city_key, hotels["default"])
            
            result = f"Hotels in {city}:\n"
            for hotel in hotel_list:
                result += f"- {hotel['name']}: ${hotel['price_usd']} USD per night\n"
            
            return result.strip()
        except Exception as e:
            return f"Error getting hotel schedule: {str(e)}"
    
    # Tool 3: Currency Converter
    def convert_currency(self, input_str: str) -> str:
        """Convert between currencies"""
        try:
            # Parse input: expected format "amount from_currency to_currency" (e.g., "100 USD to NGN")
            parts = input_str.split()
            if len(parts) < 4 or parts[2].lower() != "to":
                return "Error: Format should be 'amount from_currency to to_currency' (e.g., '100 USD to NGN')"
            
            amount = float(parts[0])
            from_currency = parts[1].upper()
            to_currency = parts[3].upper()
            
            # Exchange rates
            exchange_rates = {
                ("USD", "NGN"): 1400,
                ("NGN", "USD"): 1/1400,
                ("USD", "EUR"): 0.85,
                ("EUR", "USD"): 1/0.85,
                ("USD", "GBP"): 0.73,
                ("GBP", "USD"): 1/0.73,
            }
            
            rate_key = (from_currency, to_currency)
            if rate_key not in exchange_rates:
                return f"Error: Exchange rate for {from_currency} to {to_currency} not available"
            
            rate = exchange_rates[rate_key]
            converted_amount = amount * rate
            
            return f"{amount} {from_currency} = {converted_amount:.2f} {to_currency} (Rate: {rate})"
        except ValueError:
            return "Error: Invalid amount. Please provide a valid number."
        except Exception as e:
            return f"Error converting currency: {str(e)}"
    
    # Tool 4: RAG Query Tool
    def rag_query_tool(self, query: str) -> str:
        """Queries the RAG system for internal information"""
        try:
            # Retrieve relevant documents
            docs = self.retriever.get_relevant_documents(query)
            
            if not docs:
                return "No relevant information found in the knowledge base."
            
            # Format retrieved information
            context = "\n\n".join([
                f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
                for doc in docs[:3]  # Top 3 results
            ])
            
            return f"Retrieved Information:\n\n{context}"
        except Exception as e:
            return f"Error querying RAG system: {str(e)}"
    
    def create_tools(self) -> List[Tool]:
        """Create LangChain tools"""
        return [
            Tool(
                name="get_flight_schedule",
                func=self.get_flight_schedule,
                description="Returns flight duration and price in USD. Input should be 'origin to destination' or 'origin, destination' (e.g., 'Lagos to Nairobi')."
            ),
            Tool(
                name="get_hotel_schedule",
                func=self.get_hotel_schedule,
                description="Get hotel options for a city. Input should be a city name (e.g., 'Nairobi' or 'Lagos')."
            ),
            Tool(
                name="convert_currency",
                func=self.convert_currency,
                description="Convert between currencies. Input format: 'amount from_currency to to_currency' (e.g., '100 USD to NGN' or '920 USD to NGN')."
            ),
            Tool(
                name="RAG_Query",
                func=self.rag_query_tool,
                description="Useful for querying internal knowledge base and retrieving information from documents and past conversations. Input should be a natural language query about information you need."
            ),
        ]
    
    def create_agent(self) -> AgentExecutor:
        """Create the ReAct agent"""
        tools = self.create_tools()
        
        # ReAct prompt template
        template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
                "tool_names": ", ".join([tool.name for tool in tools])
            }
        )
        
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
    
    def run(self, user_input: str) -> str:
        """Run the agent with user input"""
        print("\n" + "="*80)
        print("CONVERSATION HISTORY:")
        print("="*80)
        
        # Display conversation history
        if self.conversation_history:
            for i, turn in enumerate(self.conversation_history, 1):
                print(f"\nTurn {i}:")
                print(f"User: {turn['user']}")
                print(f"Assistant: {turn['assistant']}")
        else:
            print("(No previous conversation)")
        
        print("\n" + "="*80)
        print("CURRENT QUERY:")
        print("="*80)
        print(f"User: {user_input}\n")
        
        print("="*80)
        print("AGENT PROCESSING:")
        print("="*80)
        
        # Create and run agent
        agent_executor = self.create_agent()
        
        try:
            result = agent_executor.invoke({"input": user_input})
            response = result.get("output", "No response generated")
        except Exception as e:
            response = f"Error: {str(e)}"
        
        print("\n" + "="*80)
        print("FINAL RESPONSE:")
        print("="*80)
        print(f"Assistant: {response}")
        print("="*80)
        
        # Save to conversation history (in memory)
        self.conversation_history.append({
            "user": user_input,
            "assistant": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save to vector store for long-term memory
        self.save_conversation_to_vectorstore(user_input, response)
        
        return response


def main():
    """Main entry point"""
    # Get user input from command line argument
    if len(sys.argv) < 2:
        print("Usage: python main.py \"Your question here\"")
        print("Example: python main.py \"What is the weather in Jos?\"")
        sys.exit(1)
    
    user_input = sys.argv[1]
    
    # Initialize and run agent
    print("Initializing RAG Agent...")
    agent = RAGAgent()
    
    # Run query
    agent.run(user_input)


if __name__ == "__main__":
    main()