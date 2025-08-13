from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
import os

load_dotenv()
print("Groq API Key:", os.getenv("GROQ_API_KEY"))


# Initialize Large Language Model (LLM) of your choice (see all models on our Models page)
llm = LLM(model="groq/llama-3.1-8b-instant")

# Create your CrewAI agents with role, main goal/objective, and backstory/personality
summarizer = Agent(
    role='Documentation Summarizer', # Agent's job title/function
    goal='Create concise summaries of technical documentation', # Agent's main objective
    backstory='Technical writer who excels at simplifying complex concepts', # Agent's background/expertise
    llm=llm, # LLM that powers your agent
    verbose=True # Show agent's thought process as it completes its task
)

translator = Agent(
    role='Technical Translator',
    goal='Translate technical documentation to other languages',
    backstory='Technical translator specializing in software documentation',
    llm=llm,
    verbose=True
)

# Define your agents' tasks
summary_task = Task(
    description='give a detailed summary about the react hook useEffect',
    expected_output="A clear, concise summary of the hook's functionality",
    agent=summarizer # Agent assigned to task
)

translation_task = Task(
    description='Translate the summary to Turkish',
    expected_output="Turkish translation of the hook documentation",
    agent=translator,
    dependencies=[summary_task] # Must run after the summary task
)

# Create crew to manage agents and task workflow
crew = Crew(
    agents=[summarizer, translator], # Agents to include in your crew
    tasks=[summary_task, translation_task], # Tasks in execution order
    verbose=True
)

result = crew.kickoff()
print(result)
