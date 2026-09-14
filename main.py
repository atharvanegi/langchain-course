from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter
from langchain_tavily import TavilySearch


model = ChatOpenRouter(
    model="openrouter/free",
    temperature=0
)

search_tool = TavilySearch()

tools = [search_tool]

agent = create_agent(
    model=model,
    tools=tools
)


def main():
    print("Hello from langchain-course!")

    result = agent.invoke({
        "messages": [
            HumanMessage(
                content="""
                Find 5 current Software Engineer jobs in Atlanta.

                For each job, give me:
                1. Company name
                2. Job title
                3. Location
                4. Direct application link

                Use the search tool and only return current openings.
                """
            )
        ]
    })

    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()