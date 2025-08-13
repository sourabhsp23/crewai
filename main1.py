import os
import json
import requests
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

# -----------------------------
# Env & LLM
# -----------------------------
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is not set.")

llm = LLM(model="groq/openai/gpt-oss-20b")

STARTUP_IDEA = os.getenv(
    "STARTUP_IDEA",
    "AI mentor for students that adapts to syllabus and personal learning style."
)

# -----------------------------
# Mock UI API helper
# -----------------------------
def generate_mock_ui_via_api(idea: str, product_spec: str = "") -> dict:
    api_url, api_key = os.getenv("MOCK_UI_API_URL"), os.getenv("DESIGN_API_KEY")
    if not api_url or not api_key:
        return {
            "status": "skipped",
            "prompt": f"Create simple mobile & desktop wireframes for:\n{idea}"
        }
    try:
        payload = {
            "idea": idea,
            "product_spec": product_spec,
            "formats": ["mobile", "desktop"]
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        resp = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}

# -----------------------------
# Agents
# -----------------------------
brand_creator = Agent(
    role='Brand Creator',
    goal='Name the startup & shape brand identity.',
    backstory='Expert in short names, taglines, and positioning.',
    llm=llm
)

product_designer = Agent(
    role='Product Designer',
    goal='Make a fast-to-build roadmap.',
    backstory='Scopes MVPs & milestones quickly.',
    llm=llm
)

marketing_lead = Agent(
    role='Marketing Lead',
    goal='Write short, catchy marketing copy.',
    backstory='Knows growth & clarity.',
    llm=llm
)

pitch_writer = Agent(
    role='Pitch Writer',
    goal='Write a 1-min verbal pitch.',
    backstory='Turns ideas into investor-friendly hooks.',
    llm=llm
)

# -----------------------------
# Tasks (shortened)
# -----------------------------
branding_task = Task(
    description=(
        f"Idea: {STARTUP_IDEA}\n"
        "Give: 5 name ideas (1-line reason each), final pick, 1-line positioning, tagline, ICP snapshot."
    ),
    expected_output="Branding pack.",
    agent=brand_creator
)

roadmap_task = Task(
    description=(
        f"Idea: {STARTUP_IDEA}\n"
        "Give: MVP (key stories), V1 milestones, brief 90-day roadmap, core architecture bullets."
    ),
    expected_output="Product plan.",
    agent=product_designer,
    dependencies=[branding_task]
)

mock_ui_task = Task(
    description=(
        f"Idea: {STARTUP_IDEA}\n"
        "List 5-6 screens with purpose, main elements, simple flow, and give JSON ui_spec."
    ),
    expected_output="UI spec JSON.",
    agent=product_designer,
    dependencies=[roadmap_task]
)

marketing_task = Task(
    description=(
        f"Idea: {STARTUP_IDEA}\n"
        "Give: landing hero+subhead, 3 bullets, 3 short social posts, short email teaser."
    ),
    expected_output="Marketing copy.",
    agent=marketing_lead,
    dependencies=[branding_task, roadmap_task]
)

pitch_task = Task(
    description=(
        f"Idea: {STARTUP_IDEA}\n"
        "Write 1-min pitch: hook, problem, solution, proof, moat, ask."
    ),
    expected_output="Pitch script.",
    agent=pitch_writer,
    dependencies=[branding_task, roadmap_task, marketing_task]
)

# -----------------------------
# Crew
# -----------------------------
crew = Crew(
    agents=[brand_creator, product_designer, marketing_lead, pitch_writer],
    tasks=[branding_task, roadmap_task, mock_ui_task, marketing_task, pitch_task],
    verbose=False
)

# -----------------------------
# Run
# -----------------------------
result = crew.kickoff()

ui_api_response = generate_mock_ui_via_api(
    idea=STARTUP_IDEA,
    product_spec="Use ui_spec JSON from UI task."
)

print("\n===== OUTPUT =====")
print(result)
