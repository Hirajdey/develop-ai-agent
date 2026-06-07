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


-->
