# streamlit.py
import os, json, requests, streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

# -----------------------------
# Env & LLM
# -----------------------------
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not set.")

llm = LLM(model="groq/llama-3.1-8b-instant")

# -----------------------------
# UI Mock API
# -----------------------------
def mock_ui_api(idea, spec=""):
    url, key = os.getenv("MOCK_UI_API_URL"), os.getenv("DESIGN_API_KEY")
    if not url or not key:
        return {"status": "skipped", "prompt": f"Create mobile & desktop wireframes for: {idea}"}
    try:
        payload = {"idea": idea, "product_spec": spec, "formats": ["mobile", "desktop"]}
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        return requests.post(url, headers=headers, data=json.dumps(payload), timeout=60).json()
    except Exception as e:
        return {"status": "error", "error": str(e)}

# -----------------------------
# Streamlit UI Config
# -----------------------------
st.set_page_config(page_title="Hackathon-in-a-Box", layout="wide")
st.title("🚀 Hackathon-in-a-Box")

startup_idea = st.text_area(
    "💡 Enter your startup idea:",
    placeholder="Example: AI mentor for students adapting to syllabus & style."
)

if st.button("Run Startup Generator"):
    if not startup_idea.strip():
        st.warning("Please enter a startup idea first.")
        st.stop()

    st.info("Running CrewAI pipeline...")

    # -----------------------------
    # Agents
    # -----------------------------
    brand = Agent(
        role='Brand & Marketing Strategist',
        goal='Create name, identity, and messaging.',
        backstory='Expert in startup naming, positioning, and concise messaging.',
        llm=llm
    )

    designer = Agent(
        role='Product Designer & Planner',
        goal='Plan MVP, roadmap, and UI structure.',
        backstory='Designs lean product plans with clear user flows.',
        llm=llm
    )

    pitcher = Agent(
        role='Pitch & Copy Writer',
        goal='Write marketing copy and investor pitch.',
        backstory='Turns ideas into persuasive, high-impact narratives.',
        llm=llm
    )

    # -----------------------------
    # Tasks
    # -----------------------------
    branding = Task(
        description=f"{startup_idea}\nGive: 5 names (1-line reason), final pick, tagline, ICP.",
        expected_output="Branding pack.",
        agent=brand
    )

    roadmap = Task(
        description=f"{startup_idea}\nGive: MVP stories, V1 milestones, 90-day plan, and 5-6 UI screens (purpose, elements, flow).",
        expected_output="Product plan with UI spec.",
        agent=designer,
        dependencies=[branding]
    )

    pitch = Task(
        description=f"{startup_idea}\nCreate: landing hero+subhead, 3 bullets, 1 email teaser, 1-min pitch (hook, problem, solution, proof, moat, ask).",
        expected_output="Marketing copy + pitch script.",
        agent=pitcher,
        dependencies=[branding, roadmap]
    )

    crew = Crew(
        agents=[brand, designer, pitcher],
        tasks=[branding, roadmap, pitch],
        verbose=False
    )

    try:
        results = crew.kickoff()

        # Display nicely
        st.success("Done!")
        st.text_area("CrewAI Output", str(results))
        # st.json(ui_resp)

        st.success("All tasks completed successfully!")

    except Exception as e:
        st.error(f"Error: {e}")
        
