<!-- Agentic AI - LangChain & LangGraph -->

<!--
uv --help
if uv not installed - pip3 install uv
next :

run "uv init" -> to initialized project (its going to create a Pyproject.toml file which is going to have the list of packages we have installed and also it will create our main.py file with some boilerplate code, some hello world code, and .python-version)

run "uv add langchain" -> This is going to install LangChain in our system. it will create few files and folder like .venv and uv.lock. the .venv basically created virtual environment for our project, so that dependencies installed for one project don't affect others.

run "uv add langchain-openai" -> this is basically to add AI Models Provider (Modal - https://docs.langchain.com/oss/python/integrations/providers/openai)

run "uv add python-dotenv" -> It's a general-purpose Python library used in almost any type of Python application to load environment variables from a .env file.

run "uv add black" -> Formats code style. It's a code formatter that automatically formats your code according to a consistent style guide.

run "uv add isort" -> Organizes imports. It's groups imports into sandard library, third-party, and local imports, making files cleaner and easier to maintain.

Next :

.gitignore ->

Next : Generate openai API Key / Google AI API Key or any others and added the key to the .env file. If you want to go with open source flatfrom like OLLAMA - to install OLLAMA we can run "uv add langchain-ollama"

Lets install OLLAMA -

from dotenv import load_dotenv
load_dotenv()

load_dotenv() loads environment variables from a .env file into the application's environment.
This allows you to securely access configuration values such as API keys, database URLs, and secrets using os.getenv() instead of hardcoding them in your source code.
It helps keep sensitive information separate from the application code and supports different configurations for different environments.

uv run python main.py -> to run a particular file


## Prompt - Prompt Template

from langchain_core.prompts import PromptTemplate

So the Prompt is someting that LLM (Large Language Models) receive as input someting which is called a prompt. so what is Prompt is a simply text input that we give the LLM and the LLM process it and return us an Output. to know more we can check the Prompt Engineeering Theory.

So now Prompt Template : A Prompt Template is a peredefined prompt with placeholders that can be filled with different values when neede. It helps create prompts in a consistent and reusable way without rewriting the same instructions every time.

As example - Tell me about {topic}. if the value of topis is python, the final prompt beccomes Tell me about Python. Prompt Templates make it easier to generate dynamic prompts and keep AI interactions organized and consistent. https://reference.langchain.com/python/langchain-core/prompts/prompt/PromptTemplate


##ChatOpenAI

from langchain_openai import ChatOpenAI

This statement imports the ChatOpenAI class from the langchain-openai package. ChatOpenAI is used to connect LangChain applications with OpenAI chat models such as GPT models. It allows you to send prompts to an OpenAI model and receive AI-generated responses within your LangChain workflow.

##LangChain Chain (we are here to write our first langchain chain)

LangChain Chain is a workflow that connects multiple components in langchain together, In a sequence where the output of one step becomes the input of the next step. Each step can be an LLM call, a prompt template, data transformatio or tool call, and it can even be another chain. and chain let us go beyond just making a single LLM prompt in response.

LangChain Chain Workflow :

```text
┌───────────────────────────────────────────┐
│ 1. User Input                             │
│ User provides a question or instruction.  │
└─────────────────┬─────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────┐
│ 2. Prompt Template                        │
│ Formats input into a structured prompt.   │
└─────────────────┬─────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────┐
│ 3. LLM Model                              │
│ Processes the prompt and generates output.│
└─────────────────┬─────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────┐
│ 4. Output Processing                      │
│ Cleans, validates, or formats the output. │
└─────────────────┬─────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────┐
│ 5. Final Response                         │
│ Returns the final result to the user.     │
└───────────────────────────────────────────┘
```

## And for an Agent / Tool Calling Workflow:

```text
┌───────────────────────────────────────────┐
│ 1. User Input                             │
│ User asks a question or gives a task.     │
└─────────────────┬─────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────┐
│ 2. Prompt Template                        │
│ Structures the request for the LLM.       │
└─────────────────┬─────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────┐
│ 3. LLM Reasoning                          │
│ Decides whether additional data is needed.│
└─────────────────┬─────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────┐
│ 4. Tool Execution                         │
│ Calls APIs, databases, or external tools. │
└─────────────────┬─────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────┐
│ 5. LLM Processing                         │
│ Analyzes tool results and drafts answer.  │
└─────────────────┬─────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────┐
│ 6. Final Response                         │
│ Returns the completed answer to the user. │
└───────────────────────────────────────────┘

```

## Building a LangChain Chain to Summarize Text

In this example, we create a simple LangChain chain that summarizes information about a person and extracts two interesting facts.

The biography is stored in the `information` variable, while a `PromptTemplate` defines the instructions that will be sent to the AI model. The `ChatOpenAI` model is then connected to the prompt template using LangChain's pipe (`|`) operator, creating a chain.

When the chain is invoked, the information is inserted into the prompt, sent to the LLM, and the generated response is returned.

### Workflow

```text
Information
    ↓
Prompt Template
    ↓
ChatOpenAI Model
    ↓
Generated Summary & Facts
```

### Code Example

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

information = """
Born and raised in a Muslim family in Rameswaram, Tamil Nadu, Kalam studied physics and aerospace engineering. He spent the next four decades as a scientist and science administrator, mainly at the Defence Research and Development Organisation (DRDO) and Indian Space Research Organisation (ISRO) and was intimately involved in India's civilian space programme and military missile development efforts.
He was known as the "Missile Man of India" for his work on the development of ballistic missile and launch vehicle technology.
He also played a pivotal organisational, technical, and political role in Pokhran-II nuclear tests in 1998.
Kalam was elected as the President of India in 2002 and was widely referred to as the "People's President".
He was awarded the Bharat Ratna, India's highest civilian honour.
Kalam passed away on 27 July 2015 while delivering a lecture at IIM Shillong.
"""

summary_template = """
Given the information {information} about a person, create:
1. A short summary
2. Two interesting facts about them
"""

summary_prompt_template = PromptTemplate(
    input_variables=["information"],
    template=summary_template
)

llm = ChatOpenAI(
    temperature=0,
    model="gpt-5.5"
)

chain = summary_prompt_template | llm

response = chain.invoke(
    {"information": information}
)

print(response.content)
```

### What Happens Internally?

1. The biography is stored in the `information` variable.
2. The `PromptTemplate` inserts the biography into the prompt.
3. The prompt is sent to the OpenAI model through `ChatOpenAI`.
4. The model generates a summary and two interesting facts.
5. The response is returned and printed to the console.

This example demonstrates the basic concept of a LangChain Chain, where the output of one component (Prompt Template) becomes the input of the next component (LLM).

-->

# Agentic AI - LangChain & LangGraph

## Prerequisites

### Install UV

Check if `uv` is already installed:

```bash
uv --help
```

If not installed:

```bash
pip3 install uv
```

---

## Project Initialization

Initialize a new project:

```bash
uv init
```

This command:

- Creates a `pyproject.toml` file to manage project dependencies.
- Creates a `main.py` file with starter boilerplate code.
- Creates a `.python-version` file.

---

## Install Required Packages

### Install LangChain

```bash
uv add langchain
```

This command:

- Installs LangChain.
- Creates a `.venv` virtual environment.
- Creates a `uv.lock` file.

The `.venv` folder provides an isolated Python environment so dependencies installed for one project do not affect other projects.

---

### Install OpenAI Integration

```bash
uv add langchain-openai
```

This package allows LangChain to connect with OpenAI models.

Reference:

https://docs.langchain.com/oss/python/integrations/providers/openai

---

### Install Python Dotenv

```bash
uv add python-dotenv
```

`python-dotenv` is a general-purpose Python library used to load environment variables from a `.env` file.

It is commonly used to manage:

- API Keys
- Database URLs
- Secrets
- Environment-specific configurations

---

### Install Black

```bash
uv add black
```

**Black** is a code formatter that automatically formats Python code according to a consistent style guide.

Benefits:

- Consistent code formatting
- Improved readability
- Reduced formatting discussions during code reviews

---

### Install isort

```bash
uv add isort
```

**isort** automatically organizes and sorts Python imports.

Benefits:

- Groups standard library imports
- Groups third-party imports
- Groups local imports
- Keeps code clean and maintainable

---

## Create .gitignore

Create a `.gitignore` file and exclude:

```gitignore
.venv/
__pycache__/
.env
.vscode/
.idea/
*.pyc
```

---

## AI Model Provider Setup

Generate an API key from your preferred AI provider:

- OpenAI
- Google AI
- Anthropic
- Others

Store the API key in a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## Using Ollama (Open Source Models)

Install Ollama integration:

```bash
uv add langchain-ollama
```

Ollama allows you to run open-source LLMs locally without relying on cloud providers.

---

## Loading Environment Variables

```python
from dotenv import load_dotenv

load_dotenv()
```

### What does `load_dotenv()` do?

- Loads environment variables from a `.env` file.
- Makes them available through `os.getenv()`.
- Prevents hardcoding sensitive values inside source code.
- Supports different configurations for different environments.

Example:

```python
import os

api_key = os.getenv("OPENAI_API_KEY")
```

---

## Running Python Files

Run a Python file using UV:

```bash
uv run python main.py
```

---

# Prompt Engineering

## Prompt

A **Prompt** is the input provided to a Large Language Model (LLM).

The model processes the prompt and generates a response.

Example:

```text
Explain FastAPI in simple terms.
```

The LLM receives the prompt and generates an appropriate answer.

---

## Prompt Template

```python
from langchain_core.prompts import PromptTemplate
```

A **Prompt Template** is a predefined prompt with placeholders that can be filled with different values when needed.

It helps create prompts in a reusable and consistent manner.

Example Template:

```text
Tell me about {topic}
```

If:

```python
topic = "Python"
```

The final prompt becomes:

```text
Tell me about Python
```

Benefits:

- Reusable
- Dynamic
- Easier maintenance
- Consistent prompt structure

Reference:

https://reference.langchain.com/python/langchain-core/prompts/prompt/PromptTemplate

---

# ChatOpenAI

```python
from langchain_openai import ChatOpenAI
```

The `ChatOpenAI` class is used to connect LangChain applications with OpenAI chat models.

It allows you to:

- Send prompts to OpenAI models
- Receive AI-generated responses
- Integrate OpenAI into LangChain workflows

Example:

```python
model = ChatOpenAI()
```

---

# LangChain Chains

## What is a Chain?

A **LangChain Chain** is a workflow that connects multiple LangChain components together.

The output of one step becomes the input of the next step.

A chain can contain:

- Prompt Templates
- LLM Calls
- Data Transformations
- Tool Calls
- Other Chains

Chains allow us to move beyond a single prompt-response interaction and build more sophisticated AI workflows.

---

## LangChain Chain Workflow

```text
┌───────────────────────────────────────────┐
│ 1. User Input                             │
│ User provides a question or instruction.  │
└─────────────────┬─────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────┐
│ 2. Prompt Template                        │
│ Formats input into a structured prompt.   │
└─────────────────┬─────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────┐
│ 3. LLM Model                              │
│ Processes the prompt and generates output.│
└─────────────────┬─────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────┐
│ 4. Output Processing                      │
│ Cleans, validates, or formats the output. │
└─────────────────┬─────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────┐
│ 5. Final Response                         │
│ Returns the final result to the user.     │
└───────────────────────────────────────────┘
```

## And for an Agent / Tool Calling Workflow:

```text
┌───────────────────────────────────────────┐
│ 1. User Input                             │
│ User asks a question or gives a task.     │
└─────────────────┬─────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────┐
│ 2. Prompt Template                        │
│ Structures the request for the LLM.       │
└─────────────────┬─────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────┐
│ 3. LLM Reasoning                          │
│ Decides whether additional data is needed.│
└─────────────────┬─────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────┐
│ 4. Tool Execution                         │
│ Calls APIs, databases, or external tools. │
└─────────────────┬─────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────┐
│ 5. LLM Processing                         │
│ Analyzes tool results and drafts answer.  │
└─────────────────┬─────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────┐
│ 6. Final Response                         │
│ Returns the completed answer to the user. │
└───────────────────────────────────────────┘

```

## Building a LangChain Chain to Summarize Text

In this example, we create a simple LangChain chain that summarizes information about a person and extracts two interesting facts.

The biography is stored in the `information` variable, while a `PromptTemplate` defines the instructions that will be sent to the AI model. The `ChatOpenAI` model is then connected to the prompt template using LangChain's pipe (`|`) operator, creating a chain.

When the chain is invoked, the information is inserted into the prompt, sent to the LLM, and the generated response is returned.

### Workflow

```text
Information
    ↓
Prompt Template
    ↓
ChatOpenAI Model
    ↓
Generated Summary & Facts
```

### Code Example

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

information = """
Born and raised in a Muslim family in Rameswaram, Tamil Nadu, Kalam studied physics and aerospace engineering. He spent the next four decades as a scientist and science administrator, mainly at the Defence Research and Development Organisation (DRDO) and Indian Space Research Organisation (ISRO) and was intimately involved in India's civilian space programme and military missile development efforts.
He was known as the "Missile Man of India" for his work on the development of ballistic missile and launch vehicle technology.
He also played a pivotal organisational, technical, and political role in Pokhran-II nuclear tests in 1998.
Kalam was elected as the President of India in 2002 and was widely referred to as the "People's President".
He was awarded the Bharat Ratna, India's highest civilian honour.
Kalam passed away on 27 July 2015 while delivering a lecture at IIM Shillong.
"""

summary_template = """
Given the information {information} about a person, create:
1. A short summary
2. Two interesting facts about them
"""

summary_prompt_template = PromptTemplate(
    input_variables=["information"],
    template=summary_template
)

llm = ChatOpenAI(
    temperature=0,
    model="gpt-5.5"
)

chain = summary_prompt_template | llm

response = chain.invoke(
    {"information": information}
)

print(response.content)
```

### What Happens Internally?

1. The biography is stored in the `information` variable.
2. The `PromptTemplate` inserts the biography into the prompt.
3. The prompt is sent to the OpenAI model through `ChatOpenAI`.
4. The model generates a summary and two interesting facts.
5. The response is returned and printed to the console.

This example demonstrates the basic concept of a LangChain Chain, where the output of one component (Prompt Template) becomes the input of the next component (LLM).
