# AI Agents + Automation

This folder contains advanced AI agent implementations that demonstrate the power of Large Language Models (LLMs) combined with tool use and automation techniques. These projects showcase how to build intelligent agents that can reason, act, and interact with various tools and systems.

## 🤖 Projects Overview

### 1. [Math Reasoning ReAct Pattern](./01_Math_Reasoning_ReAct_Patern.ipynb)
**Goal**: Create an LLM-powered agent that solves math word problems using the ReAct (Reasoning + Acting) pattern.

**Key Features**:
- Step-by-step reasoning through math problems
- Tool use for calculations (Python execution)
- Interactive problem-solving loop
- Integration with Qwen3-0.6B model

**Technologies**: Transformers, ReAct pattern, Python execution tools

### 2. [Coding Bot with Tool Use](./02_Coding_Bot_with_Tool_Use.ipynb)
**Goal**: Build a coding assistant that can write, run, and fix code automatically using LLM + Python execution tools.

**Key Features**:
- Automatic code generation and execution
- Error detection and debugging
- Iterative code improvement
- Safe code execution environment

**Technologies**: Transformers, Python execution, Code debugging

### 3. [PDF Summarizer with Agents](./03_PDF_Summarizer_with_Agents.ipynb)
**Goal**: Create an intelligent agent that loads PDFs, extracts content, and produces comprehensive summaries.

**Key Features**:
- PDF text extraction and chunking
- Intelligent content summarization
- Multi-chunk processing
- Full document synthesis

**Technologies**: PyMuPDF, Transformers, Text chunking

### 4. [AI Calendar Scheduler Agent](./04_AI_Calendar_Scheduler_Agent.ipynb)
**Goal**: Build an AI agent that can manage and schedule calendar events intelligently.

**Key Features**:
- Calendar event management
- Intelligent scheduling
- Conflict resolution
- Natural language interaction

**Technologies**: Calendar APIs, NLP, Scheduling algorithms

### 5. [Self-Correcting Essay Grader](./05_Self_Correcting_Essay_Grader.ipynb)
**Goal**: Develop an AI system that can grade essays and provide feedback with self-correction capabilities.

**Key Features**:
- Essay analysis and grading
- Feedback generation
- Self-correction mechanisms
- Educational insights

**Technologies**: NLP, Text analysis, Grading algorithms

### 6. [File Organizer Agent](./06__File_Organizer_Agent_(Desktop_AI).ipynb)
**Goal**: Create a desktop AI agent that can automatically organize files based on content and patterns.

**Key Features**:
- File system interaction
- Content-based organization
- Pattern recognition
- Automated file management

**Technologies**: File system APIs, Content analysis, Automation

### 7. [Browser Agent with Memory](./07_Browser_Agent_with_Memory.ipynb)
**Goal**: Build a browser automation agent with memory capabilities for complex web interactions.

**Key Features**:
- Web browser automation
- Memory and context retention
- Complex interaction sequences
- State management

**Technologies**: Selenium, Web automation, Memory systems

### 8. [API Router with LLM](./08_API_Router_with_LLM.ipynb)
**Goal**: Implement an intelligent API routing system that uses LLMs to direct requests to appropriate endpoints.

**Key Features**:
- Intelligent request routing
- API endpoint selection
- Request analysis
- Dynamic routing decisions

**Technologies**: API frameworks, LLM integration, Routing algorithms

### 9. [Recursive Research Agent](./09_Recursive_Research_Agent.ipynb)
**Goal**: Create an agent that can perform recursive research by following leads and building knowledge graphs.

**Key Features**:
- Recursive information gathering
- Knowledge graph construction
- Research path optimization
- Information synthesis

**Technologies**: Web scraping, Knowledge graphs, Research algorithms

### 10. [Cooking Assistant (Voice + Tools)](./10_Cooking_Assistant_(Voice_%2B_Tools).ipynb)
**Goal**: Build a voice-enabled cooking assistant that can guide users through recipes and cooking processes.

**Key Features**:
- Voice interaction
- Recipe management
- Cooking guidance
- Multi-modal interaction

**Technologies**: Speech recognition, Voice synthesis, Recipe databases

## 🛠️ Common Technologies Used

- **LLM Integration**: Hugging Face Transformers, OpenAI API
- **Tool Use**: Python execution, Web APIs, File system operations
- **Patterns**: ReAct (Reasoning + Acting), Chain-of-Thought
- **Frameworks**: LangChain, Custom agent frameworks
- **Data Processing**: Pandas, NumPy for data manipulation
- **Visualization**: Matplotlib, Seaborn for results display

## 🚀 Getting Started

### Prerequisites
```bash
pip install transformers torch pandas numpy matplotlib
pip install openai  # For OpenAI API usage
pip install langchain  # For advanced agent frameworks
```

### Running the Projects

1. **Choose a project** from the list above
2. **Open the notebook** in Jupyter or Google Colab
3. **Install dependencies** as specified in each notebook
4. **Follow the step-by-step instructions** in each project
5. **Experiment** with different parameters and configurations

### Google Colab Integration
Most notebooks include direct links to run in Google Colab:
- Click the "Open In Colab" badge at the top of each notebook
- Ensure you have access to GPU/TPU for faster processing
- Some projects may require API keys (OpenAI, etc.)

## 📚 Learning Objectives

After completing these projects, you will understand:

- **Agent Architecture**: How to design and implement AI agents
- **Tool Integration**: Connecting LLMs with external tools and APIs
- **ReAct Pattern**: Implementing reasoning and acting loops
- **Memory Systems**: Building agents with persistent memory
- **Multi-modal Interaction**: Combining text, voice, and visual inputs
- **Automation**: Creating systems that can perform complex tasks autonomously

## 🔧 Customization

Each project is designed to be easily customizable:

- **Model Selection**: Switch between different LLMs (GPT, Claude, local models)
- **Tool Integration**: Add new tools and capabilities
- **Prompt Engineering**: Modify prompts for different use cases
- **Output Formats**: Customize how agents present their results

## 🎯 Use Cases

These agents can be applied to:

- **Business Automation**: Automating repetitive tasks
- **Educational Tools**: Creating interactive learning systems
- **Content Creation**: Automated writing and summarization
- **Data Analysis**: Intelligent data processing and insights
- **Customer Service**: Automated support and assistance
- **Research**: Automated information gathering and synthesis

## 🤝 Contributing

Feel free to:
- Add new agent implementations
- Improve existing agents
- Share use cases and applications
- Report issues or suggest enhancements

## 📖 Additional Resources

- [ReAct Paper](https://arxiv.org/abs/2210.03629) - Original ReAct research
- [LangChain Documentation](https://python.langchain.com/) - Agent framework
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) - Model library
- [OpenAI API Documentation](https://platform.openai.com/docs) - API reference

---

**Note**: These projects demonstrate cutting-edge AI agent capabilities. Start with simpler projects and gradually work your way up to more complex implementations.
