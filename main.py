from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from llm_provider import get_llm

load_dotenv()


def main():
    information = """
    Born and raised in a Muslim family in Rameswaram, Tamil Nadu, Kalam studied physics and aerospace engineering.
    He spent the next four decades as a scientist and science administrator, mainly at DRDO and ISRO.
    He was known as the 'Missile Man of India' for his work on ballistic missile and launch vehicle technology.
    """

    summary_template = """
    Given the information {information} about a person, create:

    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = get_llm()
    chain = summary_prompt_template | llm 
    # And in this LCL syntax we create a chain by composing two components a prompt template and a large language

    response = chain.invoke(
        {"information": information}
    )

    print(response.content)


if __name__ == "__main__":
    main()
