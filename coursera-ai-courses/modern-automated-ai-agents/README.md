# Modern Automated AI Agents

https://learning-oreilly-com.ezproxy.bpl.org/videos/modern-automated-ai/9780135414965/9780135414965-MAAIA1_00_00_00/

## What are AI agents?

AI agents are (semi) autonomous systems that interact with environments, make decisions, and perform tasks.

They are autonomous, can make decisions, and adapt over time with feedback.

**Agents** perform specific tasks and make decisions based on environments. **LLMs* focus on understanding and generating human-like text. The LLM is a main ML engine underlying the agent.

ChatGPT is an agent on top of an LLM like GPT-4o.

Agents are **prompts** on top of LLMs designed to perform tasks & goals using tools with a set of rules, descriptions, and backstories.

* Task/goal - The thing the agent will do
* Tools - Actions the agent can perform
* Rules, descriptions, backstories - Context around the task (e.g. only speak Spanish, talk like a kindergarten teacher). These make the agent more usable for the human.
* Prompt - Consolidation of all of the above into instructions for the LLM

For example: Go to foo.com and summarize. Use this web scraper tool. Check the full site. Scroll down to load all components.

Replit is a multi-user IDE with a custom LLM and tools. Other agents use off-the-shelf LLMs with custom data and tools.

Early agents were very rule-based with limited flexibility like Alexa and Siri.

## AI agent frameworks

Frameworks exist for the creation of agents. Many of these allow creation of workflows and allow for the visualization and description of these workflows

For example LangChain and LangGraph allow building stateful, multi-actor applications with LLMs which allow having humans in the loop, statefulness, and use LangChain under the hood.

* **LangChain** - ????
* **CrewAI** - Focuses on collaborative, role-based AI agents which work on teams. It uses agent roles, dynamic task delegation, and inter-agent communication. It supports hierarchical and sequential task delegation.
* **OpenAI swarm** - Lightweight system for orchestrating multiple agents in a stateless way. Claims that improving multi-agent interactions is very important.
* **AutoGen** - Microsoft scalable distributed AI system. Available from multiple languages like C#, Python, .NET

ChatGPT is likely a single agent that runs things sequentially.

**Reliability** is a crucial trait. We use redundancy and failover mechanisms to ensure availability to deal with tool failures. Monitoring and tracking can be added on top of this.

### Using CrewAI

CrewAI defines several top-level classes like `Agent, Task, Crew`. These represent the obvious things where a `Crew` is what is used to assemble and launch (`kickoff`) the agents. Agents contain tools. Tasks might be assigned to agents, especially for sequential workflows.

In hierarchical workflows tasks do not have agents assigned but a `manager_llm` is assigned, say `ChatOpenAI from langchain_openai`. This is used to decide which agent to delegate to.

See [oreilly-ai-agents/notebooks/CrewAI_Hello_World.ipynb](oreilly-ai-agents/notebooks/CrewAI_Hello_World.ipynb) for examples.

### LangGraph

LangGraph allows you to have fine control over the processing of your query. You build up the tasks and compose them into an explicit flow chart, including conditions, that allows you to control how the system works. It can also visualize the graph.

This approach tends to be much more verbose compared to something like CrewAI since the developer makes all of the decisions rather than delegating to the LLMs to make the decisions for you.

In the [oreilly-ai-agents/notebooks/LangGraph_Hello_World.ipynb](oreilly-ai-agents/notebooks/LangGraph_Hello_World.ipynb) notebook we see examples.

## Understanding LLMs

Modern AI engineers have been working on **sequence to sequence** modelling where you input a sequence of tokens and output a sequence of tokens.

A modern history of NLP looks like

* 2001 - Neural language models
* 2013 - Encoding semantic meaning with Word2Vec
* 2014-17 - Sequence to sequence + attention
* 2017 - present - Transformers + LLMs

"Attention is all you need" introduced the transformer architecture in 2017.

Language models try to fill in the blank, effectively, by assigning a probability to optional tokens.

There are 2 types of LLMs

* **Auto-encoding** - The readers. Learn sequences by predicting tokens given past and future context. Great for classification, embedding + retrieval tasks. E.g. BERT, XLNET, RoBERTa, sBERT
* **Auto-regressive** - The writers. Predict a future token given either past or future context, but not both. Can generate text. Can read text but must be larger than auto-encoding for similar functionality. E.g. GPT, Llama, Anthropic's Claude, most LLMs/"gen AI".

## Agentic tools

Tools run with a set of inputs and return an output. To implement custom tools in CrewAI you can use the `@tool` decorator from `crewai_tools`. The tool function docstring is leveraged by the agent to decide when to use the tool. `BaseTool` can also be used instead of the decorator.

[oreilly-ai-agents/notebooks/CrewAI_Hello_World.ipynb](oreilly-ai-agents/notebooks/CrewAI_Hello_World.ipynb) has an example of this in the "Buildig a Search Agent with a Custom Tool" section.

Note that when allowing LLMs to make decisions and delegate, they can make wrong or seemingly nonsensical decisions.

## Actions

- [ ] Vector database like Chroma
- [ ] Read about transformer architecture
- [ ] Read about RAG
