from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

load_dotenv()

from langchain_ollama import ChatOllama


def main():
    print("Hello from langchain-course!")

    information = """
    Virat Kohli (born 5 November 1988) is an Indian international cricketer
    and former all-format captain of the Indian national cricket team.
    """

    summary_template = """
    Given the information {information} about a person, I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    model = init_chat_model(
        "auto",
        model_provider="openrouter",
        temperature=0
    )

    # model = init_chat_model(
    #     "gemma3:270m",
    #     model_provider="ollama"
    # )

    chain = summary_prompt_template | model

    response = chain.invoke({
        "information": information
    })

    print(response)
    print(response.content)


if __name__ == "__main__":
    main()