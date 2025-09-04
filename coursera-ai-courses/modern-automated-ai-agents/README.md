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


