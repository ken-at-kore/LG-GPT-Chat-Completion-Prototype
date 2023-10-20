# LG AiBot Prototpye

This prototype demonstrates a chatbot experience with an LG sales associate personality. This bot leverages the ChatGPT API Function Calling capabilities to give the AI the ability to query a product database to load product data into its LLM context window. This lets the AI independently retrieve the data it needs to accurately answer the user's LG product questions.

## ChatGPT Function Calling
- Explanation (OpenAI blog post): https://openai.com/blog/function-calling-and-other-api-updates

- Documentation (OpenAI): https://openai.com/blog/function-calling-and-other-api-updates

- A Tutorial Guide to Using The Function Call Feature of OpenAI's ChatGPT API (blog post): https://codeconfessions.substack.com/p/creating-chatgpt-plugins-using-the

## Streamlit web framework
The prototype uses a service and Python-based web framework called Streamlit. https://streamlit.io

- Streamlit tutorial: Build conversational apps: https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps

## AiBot Framework architecture & features
The prototype is based on an AiBot framework developed by Ken Grafals (Kore.ai). It implements the following architectures and design patterns:

### MRKL
MRKL systems aim to create a hybrid architecture that combines the strengths of large language models with external knowledge sources and discrete reasoning capabilities. The goal is to enhance the performance and reliability of AI systems by integrating neural networks with symbolic reasoning. Developed by AI21 Labs, MRKL strives to usher in a new era of more robust and versatile AI applications. [Research paper](https://arxiv.org/abs/2205.00445)

### ReAct
ReAct synergizes reasoning and acting capabilities in Large Language Models (LLMs), traditionally studied separately, to enhance interactive decision-making and language understanding tasks. It interleaves reasoning traces and task-specific actions, allowing models to interact with external sources and improve human interpretability and task-solving effectiveness. ReAct prompts, comprising few-shot task-solving trajectories with human-written text reasoning traces and actions, are intuitive to design and achieve state-of-the-art performance across various tasks like question answering and online shopping​. [Research paper](https://react-lm.github.io)

### Chain of thought
"Chain of Thought" is a technique that improves the reasoning capabilities of large language models by guiding them through a series of intermediate reasoning steps. This approach allows the model to systematically work through complex problems, enhancing its problem-solving abilities. Developed by Google Research's Brain Team, the method demonstrates that large language models can perform more effective reasoning when structured in this way. [Research paper](https://arxiv.org/abs/2201.11903)

### Conversation summary buffer
While the AiBot framework does not use LangChain, it does implement a feature akin to LangChain's Conversation Summary Buffer. This feature maintains a buffer of recent interactions, but unlike merely discarding old interactions, it compiles them into a summary, utilizing both the buffer and the summary for various purposes. [LangChain Conversation Summary Buffer](https://python.langchain.com/docs/modules/memory/types/summary_buffer)