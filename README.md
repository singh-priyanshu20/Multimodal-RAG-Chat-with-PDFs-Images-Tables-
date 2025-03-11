# Multi-modal Retrieval-Augmented Generation (RAG) with LangChain

## 📌 Overview
This project implements a **Multi-modal Retrieval-Augmented Generation (RAG) system** that extracts **text, images, and tables** from PDFs to generate context-aware responses. It leverages **LangChain, Vector Databases, LLaMA-3.1-8B-Instant, and GPT-4o-mini** for information retrieval and response generation.

## 🚀 Features
- Extracts **text, tables, and images** from PDFs.
- Uses a **Vector Database (FAISS/ChromaDB)** for efficient retrieval.
- Implements **LangChain** for document parsing and querying.
- Uses **LLaMA-3.1-8B-Instant** and **GPT-4o-mini** for response generation.
- Fully functional **end-to-end pipeline** for AI-powered document analysis.

## 🛠️ Installation & Setup
### **1. Clone the Repository**
```sh
 git clone https://github.com/your-username/your-repo.git
 cd your-repo
```

### **2. Install Dependencies**
```sh
# For Linux
sudo apt-get install poppler-utils tesseract-ocr libmagic-dev

# Install Python dependencies
pip install -Uq "unstructured[all-docs]" pillow lxml pillow
pip install -Uq chromadb tiktoken
pip install -Uq langchain langchain-community langchain-openai langchain-groq
pip install -Uq python_dotenv
```

### **3. Set API Keys**
Create an **.env** file and add the following:
```sh
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_TRACING_V2=true
```

## 📖 Usage
### **1. Extract Data from PDFs**
```python
from unstructured.partition.pdf import partition_pdf
chunks = partition_pdf(filename='document.pdf', infer_table_structure=True, strategy="hi_res")
```

### **2. Summarize Text and Tables**
```python
from langchain_groq import ChatGroq
model = ChatGroq(model="llama-3.1-8b-instant")
text_summaries = model.batch([chunk.text for chunk in chunks])
```

### **3. Summarize Images**
```python
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini")
image_summaries = model.batch(images)
```

### **4. Load Summaries into Vector Database**
```python
from langchain.vectorstores import Chroma
vectorstore = Chroma(collection_name="multi_modal_rag")
vectorstore.add_documents(text_summaries + image_summaries)
```

### **5. Query the RAG System**
```python
response = chain.invoke("What is the attention mechanism?")
print(response)
```

## 🏗️ System Architecture
1️⃣ **Text Preprocessing:** Extracts and chunks text, tables, and images from PDFs.  
2️⃣ **Embedding Generation:** Converts chunks into vector embeddings using **FAISS/ChromaDB**.  
3️⃣ **Retrieval:** Uses a **multi-vector retriever** to fetch the most relevant chunks.  
4️⃣ **Generation:** Passes retrieved context to **LLaMA-3.1-8B-Instant** or **GPT-4o-mini** for final response.  

## 🛠️ Technologies Used
- **Python** 🐍  
- **LangChain** (Document processing & retrieval)  
- **Vector Databases** (FAISS/ChromaDB)  
- **LLaMA-3.1-8B-Instant, GPT-4o-mini** (Response generation)  
- **Unstructured Library** (PDF processing)  

## 🔮 Future Enhancements
- 🌐 **Integrate Streamlit for a Web UI**  
- 🏗️ **Support for multi-document search**  
- 🚀 **Optimize retrieval accuracy using BM25**  
