# Natural Language Processing (NLP)

This folder contains comprehensive Natural Language Processing projects that demonstrate various techniques for text analysis, understanding, and generation. These projects cover everything from basic text processing to advanced transformer-based models and retrieval-augmented generation systems.

## 🗣️ Projects Overview

### 1. [Named Entity Recognition with spaCy](./Named_Entity_Recognition_with_spaCy.ipynb)
**Goal**: Extract named entities (like PERSON, ORG, DATE, GPE) from text using spaCy's built-in NER model.

**Key Features**:
- Pre-trained spaCy NER model (en_core_web_sm)
- Entity extraction and classification
- Entity visualization with spaCy's displacy
- Entity grouping by type
- Support for multiple entity types (PERSON, ORG, GPE, DATE, LOC, etc.)

**Technologies**: spaCy, NER, Text processing, Entity visualization

**Example Output**: Extracts entities like "Elon Musk" (PERSON), "Tesla, Inc." (ORG), "South Africa" (GPE)

### 2. [Resume (PDF) Parser](./Resume_(PDF)Parser.ipynb)
**Goal**: Build an intelligent parser that reads resumes and extracts structured details like contact info, skills, and job history using text extraction and pattern recognition.

**Key Features**:
- PDF text extraction using pdfplumber
- Structured data extraction (contact info, skills, experience)
- Pattern recognition for different resume formats
- Data validation and cleaning
- Export to structured formats (JSON, CSV)

**Technologies**: pdfplumber, spaCy, Regular expressions, Data extraction

### 3. [Semantic Search with Sentence Transformers](./Semantic_Search_with_Sentence_Transformers.ipynb)
**Goal**: Implement semantic search capabilities using sentence transformers to find relevant documents based on meaning rather than exact keyword matches.

**Key Features**:
- Sentence transformer models for embeddings
- Vector similarity search
- Semantic document retrieval
- Query expansion and refinement
- Performance optimization for large document collections

**Technologies**: Sentence Transformers, FAISS, Vector databases, Semantic search

### 4. [Simple RAG System using ChromaDB and Transformers](./Simple_RAG_System_using_ChromaDB_and_Transformers.ipynb)
**Goal**: Build a Retrieval-Augmented Generation (RAG) system that combines document retrieval with text generation for accurate, context-aware responses.

**Key Features**:
- Document chunking and embedding
- ChromaDB vector database integration
- Context retrieval and ranking
- LLM integration for answer generation
- End-to-end RAG pipeline

**Technologies**: ChromaDB, Transformers, RAG, Vector databases, LLM integration

## 🛠️ Common Technologies Used

- **NLP Libraries**: spaCy, NLTK, Transformers, Sentence Transformers
- **Text Processing**: Regular expressions, Text preprocessing, Tokenization
- **Vector Databases**: ChromaDB, FAISS, Pinecone
- **PDF Processing**: pdfplumber, PyMuPDF, pdf2image
- **Data Processing**: Pandas, NumPy for data manipulation
- **Visualization**: Matplotlib, Seaborn, spaCy displacy

## 🚀 Getting Started

### Prerequisites
```bash
pip install spacy pandas numpy matplotlib
pip install sentence-transformers  # For semantic search
pip install chromadb  # For vector databases
pip install pdfplumber  # For PDF processing
pip install transformers  # For transformer models

# Download spaCy models
python -m spacy download en_core_web_sm
```

### Running the Projects

1. **Choose a project** from the list above
2. **Open the notebook** in Jupyter or Google Colab
3. **Install dependencies** as specified in each notebook
4. **Download required models** (spaCy models, transformer models)
5. **Follow the step-by-step implementation**
6. **Experiment** with different texts and parameters

### Google Colab Integration
Most notebooks include direct links to run in Google Colab:
- Click the "Open In Colab" badge at the top of each notebook
- Some projects may require GPU for transformer models
- Ensure proper model downloads and installations

## 📊 Key NLP Concepts Covered

### Named Entity Recognition (NER)
- **Entity Types**: PERSON, ORG, GPE, DATE, LOC, MONEY, PERCENT, etc.
- **Model Training**: Using spaCy's pre-trained models
- **Custom Entities**: Training models for domain-specific entities
- **Evaluation**: Measuring NER performance with metrics

### Document Processing
- **PDF Extraction**: Converting PDFs to text while preserving structure
- **Text Cleaning**: Removing noise and standardizing format
- **Chunking**: Breaking documents into manageable pieces
- **Metadata Extraction**: Identifying document structure and key information

### Semantic Search
- **Embeddings**: Converting text to vector representations
- **Similarity Metrics**: Cosine similarity, Euclidean distance
- **Vector Databases**: Efficient storage and retrieval of embeddings
- **Query Processing**: Understanding user intent and context

### Retrieval-Augmented Generation (RAG)
- **Document Indexing**: Creating searchable document collections
- **Context Retrieval**: Finding relevant information for queries
- **Answer Generation**: Using LLMs with retrieved context
- **Evaluation**: Measuring RAG system performance

## 🎯 Learning Objectives

After completing these projects, you will understand:

- **Text Preprocessing**: Cleaning and preparing text data
- **Entity Recognition**: Identifying and classifying named entities
- **Document Analysis**: Extracting structured information from unstructured text
- **Vector Representations**: Converting text to numerical vectors
- **Semantic Understanding**: Going beyond keyword matching
- **RAG Systems**: Combining retrieval and generation for accurate responses
- **NLP Pipelines**: End-to-end text processing workflows

## 🔧 Model Architectures and Techniques

- **spaCy Models**: Pre-trained NER and dependency parsing
- **Sentence Transformers**: BERT-based sentence embeddings
- **Transformer Models**: BERT, RoBERTa, T5 for various NLP tasks
- **Vector Databases**: ChromaDB, FAISS for efficient similarity search
- **RAG Pipelines**: Document retrieval + LLM generation

## 📈 Performance Optimization

Each project includes techniques for:
- **Model Selection**: Choosing appropriate models for specific tasks
- **Text Preprocessing**: Optimizing text cleaning and normalization
- **Vector Search**: Efficient similarity computation
- **Memory Management**: Handling large document collections
- **Batch Processing**: Processing multiple documents efficiently

## 🎨 Applications

These NLP techniques can be applied to:

- **Document Processing**: Automated resume parsing, contract analysis
- **Information Retrieval**: Semantic search engines, question answering
- **Content Analysis**: Sentiment analysis, topic modeling
- **Data Extraction**: Structured data from unstructured text
- **Chatbots**: Intelligent conversational agents
- **Knowledge Management**: Building searchable knowledge bases
- **Legal Tech**: Contract analysis, legal document processing

## 📚 Datasets and Resources

- **spaCy Models**: Pre-trained models for multiple languages
- **Sentence Transformers**: Hugging Face model hub
- **Sample Documents**: Resume samples, PDF documents
- **Evaluation Datasets**: Standard NLP benchmarks

## 🤝 Contributing

Feel free to:
- Add new NLP projects and techniques
- Improve existing implementations
- Share performance optimizations
- Report issues or suggest enhancements
- Contribute new datasets or use cases

## 📖 Additional Resources

- [spaCy Documentation](https://spacy.io/) - NLP library documentation
- [Sentence Transformers](https://www.sbert.net/) - Semantic search library
- [ChromaDB Documentation](https://docs.trychroma.com/) - Vector database
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) - Model library
- [RAG Paper](https://arxiv.org/abs/2005.11401) - Retrieval-Augmented Generation research

## 🔍 Troubleshooting

Common issues and solutions:
- **Model Downloads**: Ensure proper spaCy model installation
- **Memory Issues**: Use batch processing for large documents
- **PDF Processing**: Handle different PDF formats and encodings
- **Vector Search**: Optimize similarity thresholds for better results

## 🚀 Advanced Topics

For those looking to extend these projects:
- **Custom NER Models**: Training domain-specific entity recognizers
- **Multi-language Support**: Extending to non-English languages
- **Real-time Processing**: Building streaming NLP pipelines
- **Model Fine-tuning**: Adapting pre-trained models for specific domains

---

**Note**: Start with Named Entity Recognition to understand basic NLP concepts, then progress to more complex systems like RAG. Each project builds upon fundamental NLP principles and modern transformer architectures.
