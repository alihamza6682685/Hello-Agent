from dotenv import load_dotenv
import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load Gemini API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
print("API Key loaded:", bool(api_key))

# Create Gemini client
external_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Wrap the client in a model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Create RunConfig that uses your model
config = RunConfig(
    model=model,
    tracing_disabled=True
)

# Create the agent
agent = Agent(
    name="hello_agent",
    instructions="You are a concise assistant that greets the user in a single short sentence."
)

# Run the agent with the custom config
result = Runner.run_sync(agent, "Say hello in one short sentence.", run_config=config)

# Print the result
print(result.final_output)
