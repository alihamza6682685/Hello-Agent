from dotenv import load_dotenv
import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

def run():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    print("API Key loaded:", bool(api_key))

    # Create Gemini client
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    # Wrap client in model
    model = OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=client
    )

    # RunConfig for agent
    config = RunConfig(
        model=model,
        tracing_disabled=True
    )

    # Create agent
    agent = Agent(
        name="hello_agent",
        instructions="You are a concise assistant that greets the user in a single short sentence."
    )

    # Run agent
    result = Runner.run_sync(agent, "Say hello in one short sentence.", run_config=config)
    print(result.final_output)
