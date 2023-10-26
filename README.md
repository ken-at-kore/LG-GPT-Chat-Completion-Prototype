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

### MRKL System / LLM Agent
MRKL systems aim to create a hybrid architecture that combines the strengths of large language models with external knowledge sources and discrete reasoning capabilities. The goal is to enhance the performance and reliability of AI systems by integrating neural networks with symbolic reasoning. Developed by AI21 Labs, MRKL strives to usher in a new era of more robust and versatile AI applications. [Research paper](https://arxiv.org/abs/2205.00445)

### Chain of thought & ReAct
Chain of Thought (CoT) enhances LLMs by guiding them through structured reasoning steps for better problem-solving. ReAct builds on this by interleaving reasoning with task-specific actions, enabling interaction with external sources. It uses chain of thought reasoning traces to enhance decision making capabilities. [CoT research paper. ](https://arxiv.org/abs/2201.11903) [ReAct Research paper.](https://react-lm.github.io)

While the AiBot framework has this capability, it is currently disabled in the LG application. The current version of the chain of thought feature doesn't seem to improve decision making.

### Conversation summary buffer
The AiBot framework uses the LangChain memory Python module in order to implement the LangChain Conversation Summary Buffer Memory. This feature maintains a buffer of recent interactions, but instead of discarding old interactions, it compiles them into a summary with an LLM, utilizing both the buffer and the summary for conversation context. [LangChain Conversation Summary Buffer](https://python.langchain.com/docs/modules/memory/types/summary_buffer)