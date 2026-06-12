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


# Switching Between Multiple LLM Providers

As AI applications grow, it is common to support multiple Large Language Model (LLM) providers such as OpenAI, Google Gemini, and Ollama. Instead of hardcoding a specific provider inside the application, we can create a centralized LLM provider module and switch providers using environment variables.

This approach makes the application more flexible, maintainable, and easier to test across different models.

---

## Benefits

* Easily switch between OpenAI, Gemini, and Ollama.
* No code changes required when changing providers.
* Keeps provider-specific logic separate from business logic.
* Makes the application more scalable and maintainable.

---

## Project Structure

```text
develop-ai-agent/
│
├── .env
├── main.py
├── llm_provider.py
├── pyproject.toml
├── uv.lock
└── .venv/
```

---

## Install Required Packages

```bash
uv add langchain-openai
uv add langchain-google-genai
uv add langchain-ollama
```

---

## Environment Configuration

Create a `.env` file:

```env
LLM_PROVIDER=openai

OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
```

To switch providers:

```env
LLM_PROVIDER=gemini
```

or

```env
LLM_PROVIDER=ollama
```

---

## Creating the LLM Provider Module

Create a file named `llm_provider.py`.

```python
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "openai")

    if provider == "openai":
        return ChatOpenAI(
            model="gpt-5.5",
            temperature=0,
        )

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
        )

    if provider == "ollama":
        return ChatOllama(
            model="llama3.2",
            temperature=0,
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider}"
    )
```

---

## Using the Provider in Main Application

```python
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from llm_provider import get_llm

load_dotenv()


def main():
    information = """
    Born and raised in a Muslim family in Rameswaram, Tamil Nadu,
    Kalam studied physics and aerospace engineering.
    """

    summary_template = """
    Given the information {information}
    about a person, create:

    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = get_llm()

    chain = summary_prompt_template | llm

    response = chain.invoke(
        {"information": information}
    )

    print(response.content)


if __name__ == "__main__":
    main()
```

---

## Workflow

```text
                .env
                  │
                  ▼
        LLM_PROVIDER=openai
                  │
                  ▼
         get_llm() Function
                  │
       ┌──────────┼──────────┐
       │          │          │
       ▼          ▼          ▼
    OpenAI     Gemini     Ollama
       │          │          │
       └──────────┼──────────┘
                  │
                  ▼
         LangChain Chain
                  │
                  ▼
           AI Response
```

---

## How It Works

1. The application reads the selected provider from the `.env` file.
2. The `get_llm()` function creates the appropriate LLM instance.
3. The LangChain chain uses the returned model.
4. Changing the provider only requires updating the `.env` file.
5. No modifications are needed in the application logic.

---

## Example Provider Switching

### OpenAI

```env
LLM_PROVIDER=openai
```

### Google Gemini

```env
LLM_PROVIDER=gemini
```

### Ollama

```env
LLM_PROVIDER=ollama
```

This design follows the Separation of Concerns principle by keeping LLM configuration isolated from the rest of the application.



# Additional Setup for Ollama

## What is Ollama?

Ollama is a platform that allows you to run Large Language Models (LLMs) locally on your machine. Unlike cloud-based providers such as OpenAI or Google Gemini, Ollama does not require an API key and can run models directly on your computer.

Commonly used models include:

* Llama 3.2
* Qwen 3
* Mistral
* Gemma

---

# Install Ollama

Download and install Ollama from:

https://ollama.com/download

After installation, verify that Ollama is available on your system.

```bash
ollama --version
```

Expected output:

```text
ollama version x.x.x
```

---

# Download a Model

Before using Ollama with LangChain, you must download at least one model.

Example:

```bash
ollama pull llama3.2
```

For coding and AI Agent development:

```bash
ollama pull qwen3
```

---

# List Installed Models

To see all downloaded models:

```bash
ollama list
```

Example output:

```text
NAME        ID        SIZE
llama3.2    xxxxxx    2.0 GB
qwen3       xxxxxx    5.2 GB
```

---

# Test a Model

Run the model directly from the terminal:

```bash
ollama run llama3.2
```

Example:

```text
>>> Who was A.P.J. Abdul Kalam?
```

Exit the chat session:

```text
/bye
```

or

```text
Ctrl + C
```

---

# Install LangChain Ollama Integration

Add the LangChain Ollama package to your project:

```bash
uv add langchain-ollama
```

---

# Configure Environment Variables

Unlike OpenAI or Gemini, Ollama does not require an API key.

Example:

```env
LLM_PROVIDER=ollama
```

---

# Using Ollama in LangChain

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)
```

---

# Using Ollama with Multiple Providers

Example `.env` configuration:

```env
LLM_PROVIDER=ollama
```

Switching to OpenAI:

```env
LLM_PROVIDER=openai
```

Switching to Gemini:

```env
LLM_PROVIDER=gemini
```

No code changes are required if your application uses a centralized `get_llm()` provider function.

---

# Recommended Models

## Llama 3.2

```bash
ollama pull llama3.2
```

Good for:

* Learning LangChain
* Prompt Engineering
* Basic AI Agents
* General-purpose tasks

---

## Qwen 3

```bash
ollama pull qwen3
```

Good for:

* Coding
* Tool Calling
* AI Agents
* LangGraph
* Reasoning Tasks

---

## Mistral

```bash
ollama pull mistral
```

Good for:

* Fast inference
* Lightweight local deployments

---

# Ollama Workflow

```text
Install Ollama
        ↓
Download Model
        ↓
Verify Installation
        ↓
Install langchain-ollama
        ↓
Configure LLM_PROVIDER=ollama
        ↓
Create ChatOllama Instance
        ↓
Use Inside LangChain Chains
        ↓
Generate Responses
```

---

# Benefits of Ollama

* Runs completely locally
* No API key required
* No usage costs
* Supports multiple open-source models
* Easy integration with LangChain
* Great for learning AI Agents and LangGraph

---

# Summary

Ollama provides a simple way to run open-source LLMs locally and integrate them with LangChain. By combining Ollama with a provider abstraction layer, you can easily switch between OpenAI, Gemini, and local models without modifying your application code.


# VS Code Debug Setup (Python + uv)

Before starting debugging, open the actual project folder in VS Code (the folder containing `pyproject.toml`, `.venv`, and source files). Make sure the uv virtual environment is created and select the `.venv` Python interpreter in VS Code.

Configure `.vscode/settings.json` to point to the project virtual environment and create a `launch.json` debug configuration. Verify that VS Code debugger is using `.venv\Scripts\python.exe` instead of the global Python installation.

After setup, add breakpoints and start debugging with `F5`. The debugger should run with the same environment as:

```bash
uv run python main.py
```


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

# Switching Between Multiple LLM Providers

As AI applications grow, it is common to support multiple Large Language Model (LLM) providers such as OpenAI, Google Gemini, and Ollama. Instead of hardcoding a specific provider inside the application, we can create a centralized LLM provider module and switch providers using environment variables.

This approach makes the application more flexible, maintainable, and easier to test across different models.

---

## Benefits

- Easily switch between OpenAI, Gemini, and Ollama.
- No code changes required when changing providers.
- Keeps provider-specific logic separate from business logic.
- Makes the application more scalable and maintainable.

---

## Project Structure

```text
develop-ai-agent/
│
├── .env
├── main.py
├── llm_provider.py
├── pyproject.toml
├── uv.lock
└── .venv/
```

---

## Install Required Packages

```bash
uv add langchain-openai
uv add langchain-google-genai
uv add langchain-ollama
```

---

## Environment Configuration

Create a `.env` file:

```env
LLM_PROVIDER=openai

OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
```

To switch providers:

```env
LLM_PROVIDER=gemini
```

or

```env
LLM_PROVIDER=ollama
```

---

## Creating the LLM Provider Module

Create a file named `llm_provider.py`.

```python
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def get_llm():
    provider = os.getenv("LLM_PROVIDER", "openai")

    if provider == "openai":
        return ChatOpenAI(
            model="gpt-5.5",
            temperature=0,
        )

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
        )

    if provider == "ollama":
        return ChatOllama(
            model="llama3.2",
            temperature=0,
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider}"
    )
```

---

## Using the Provider in Main Application

```python
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from llm_provider import get_llm

load_dotenv()


def main():
    information = """
    Born and raised in a Muslim family in Rameswaram, Tamil Nadu,
    Kalam studied physics and aerospace engineering.
    """

    summary_template = """
    Given the information {information}
    about a person, create:

    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = get_llm()

    chain = summary_prompt_template | llm

    response = chain.invoke(
        {"information": information}
    )

    print(response.content)


if __name__ == "__main__":
    main()
```

---

## Workflow

```text
                .env
                  │
                  ▼
        LLM_PROVIDER=openai
                  │
                  ▼
         get_llm() Function
                  │
       ┌──────────┼──────────┐
       │          │          │
       ▼          ▼          ▼
    OpenAI     Gemini     Ollama
       │          │          │
       └──────────┼──────────┘
                  │
                  ▼
         LangChain Chain
                  │
                  ▼
           AI Response
```

---

## How It Works

1. The application reads the selected provider from the `.env` file.
2. The `get_llm()` function creates the appropriate LLM instance.
3. The LangChain chain uses the returned model.
4. Changing the provider only requires updating the `.env` file.
5. No modifications are needed in the application logic.

---

## Example Provider Switching

### OpenAI

```env
LLM_PROVIDER=openai
```

### Google Gemini

```env
LLM_PROVIDER=gemini
```

### Ollama

```env
LLM_PROVIDER=ollama
```

This design follows the Separation of Concerns principle by keeping LLM configuration isolated from the rest of the application.

# Additional Setup for Ollama

## What is Ollama?

Ollama is a platform that allows you to run Large Language Models (LLMs) locally on your machine. Unlike cloud-based providers such as OpenAI or Google Gemini, Ollama does not require an API key and can run models directly on your computer.

Commonly used models include:

- Llama 3.2
- Qwen 3
- Mistral
- Gemma

---

# Install Ollama

Download and install Ollama from:

https://ollama.com/download

After installation, verify that Ollama is available on your system.

```bash
ollama --version
```

Expected output:

```text
ollama version x.x.x
```

---

# Download a Model

Before using Ollama with LangChain, you must download at least one model.

Example:

```bash
ollama pull llama3.2
```

For coding and AI Agent development:

```bash
ollama pull qwen3
```

---

# List Installed Models

To see all downloaded models:

```bash
ollama list
```

Example output:

```text
NAME        ID        SIZE
llama3.2    xxxxxx    2.0 GB
qwen3       xxxxxx    5.2 GB
```

---

# Test a Model

Run the model directly from the terminal:

```bash
ollama run llama3.2
```

Example:

```text
>>> Who was A.P.J. Abdul Kalam?
```

Exit the chat session:

```text
/bye
```

or

```text
Ctrl + C
```

---

# Install LangChain Ollama Integration

Add the LangChain Ollama package to your project:

```bash
uv add langchain-ollama
```

---

# Configure Environment Variables

Unlike OpenAI or Gemini, Ollama does not require an API key.

Example:

```env
LLM_PROVIDER=ollama
```

---

# Using Ollama in LangChain

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)
```

---

# Using Ollama with Multiple Providers

Example `.env` configuration:

```env
LLM_PROVIDER=ollama
```

Switching to OpenAI:

```env
LLM_PROVIDER=openai
```

Switching to Gemini:

```env
LLM_PROVIDER=gemini
```

No code changes are required if your application uses a centralized `get_llm()` provider function.

---

# Recommended Models

## Llama 3.2

```bash
ollama pull llama3.2
```

Good for:

- Learning LangChain
- Prompt Engineering
- Basic AI Agents
- General-purpose tasks

---

## Qwen 3

```bash
ollama pull qwen3
```

Good for:

- Coding
- Tool Calling
- AI Agents
- LangGraph
- Reasoning Tasks

---

## Mistral

```bash
ollama pull mistral
```

Good for:

- Fast inference
- Lightweight local deployments

---

# Ollama Workflow

```text
Install Ollama
        ↓
Download Model
        ↓
Verify Installation
        ↓
Install langchain-ollama
        ↓
Configure LLM_PROVIDER=ollama
        ↓
Create ChatOllama Instance
        ↓
Use Inside LangChain Chains
        ↓
Generate Responses
```

---

# Benefits of Ollama

- Runs completely locally
- No API key required
- No usage costs
- Supports multiple open-source models
- Easy integration with LangChain
- Great for learning AI Agents and LangGraph

---

# Summary

Ollama provides a simple way to run open-source LLMs locally and integrate them with LangChain. By combining Ollama with a provider abstraction layer, you can easily switch between OpenAI, Gemini, and local models without modifying your application code.

# VS Code Debug Setup (Python + uv)

Before starting debugging, open the actual project folder in VS Code (the folder containing `pyproject.toml`, `.venv`, and source files). Make sure the uv virtual environment is created and select the `.venv` Python interpreter in VS Code.

Configure `.vscode/settings.json` to point to the project virtual environment and create a `launch.json` debug configuration. Verify that VS Code debugger is using `.venv\Scripts\python.exe` instead of the global Python installation.

Create a .vscode folder inside the project root to store VS Code-specific configurations :

```
    └── .vscode/
        ├── settings.json
        └── launch.json

    settings.json:

    {
    "python.defaultInterpreterPath": "${workspaceFolder}\\.venv\\Scripts\\python.exe"
    }

    launch.json:

    {
        "version": "0.2.0",
        "configurations": [
            {
            "name": "Debug main.py",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/main.py",
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env"
            }
        ]
    }
```

After setup, add breakpoints and start debugging with `F5`. The debugger should run with the same environment as:

```bash
uv run python main.py
```
