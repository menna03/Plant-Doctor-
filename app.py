"""Plant Doctor – Streamlit UI.

Run with:  streamlit run app.py
"""
import json
import os
import time
from datetime import datetime

import streamlit as st
from google import genai
from google.genai import errors, types
from PIL import Image

# --- Configuration -----------------------------------------------------------
TEXT_MODEL = "gemma-4-26b-a4b-it"
SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
REQUEST_TIMEOUT_MS = 300_000
MAX_IMAGE_SIDE = 1024
MAX_RETRIES = 3

SEVERITY_LEVELS = ("Healthy", "Mild", "Moderate", "Severe", "Uncertain")
SEVERITY_COLORS = {
    "Healthy": "green",
    "Mild": "blue",
    "Moderate": "orange",
    "Severe": "red",
    "Uncertain": "gray",
}

NEGATIVE_PROMPT = "wilted, drooping, dead leaves, text, watermark, cartoon"
RECOVERY_STAGES = [
    # (label, prompt, strength)
    ("Day 1 (today)", None, None),  # original photo, unchanged
    ("Week 1",
     "the same houseplant in the same pot, leaves slightly firmer and less droopy, "
     "early signs of recovery, photorealistic", 0.35),
    ("Week 2",
     "the same houseplant in the same pot, mostly upright with firm green leaves, "
     "clearly recovering, photorealistic", 0.50),
    ("Final (4 weeks)",
     "the same houseplant in the same pot, healthy upright firm green leaves, "
     "fully recovered, photorealistic", 0.65),
]

ANALYSIS_PROMPT = """
You are Plant Doctor, an AI-powered plant health assistant.

Examine the uploaded plant image and return your analysis
using the following JSON structure:

{
    "plant_identification": {
        "common_name": "",
        "scientific_name": "",
        "identification_confidence": ""
    },
    "visible_symptoms": [],
    "possible_health_problems": [
        {"problem": "", "reason": "", "likelihood": ""}
    ],
    "environmental_factors": [],
    "recommended_actions": [],
    "monitoring_advice": [],
    "overall_assessment": ""
}

Rules:
- Identify the plant only if the image provides enough evidence.
- Describe only symptoms visible in the photograph.
- Distinguish observations from possible explanations.
- Never claim a definitive disease diagnosis.
- Do not invent symptoms.
- Explain uncertainty.
- Give practical and safe plant-care recommendations.
- Return valid JSON only, without Markdown code fences.
"""

CARE_PLAN_PROMPT = """
You are an experienced plant-care assistant.

Create a practical care plan based on this plant health report:

{report}

Return valid JSON with this structure:

{{
    "watering": {{"recommendation": "", "frequency": "", "warning": ""}},
    "sunlight": {{"recommendation": "", "ideal_conditions": ""}},
    "soil_and_drainage": "",
    "pest_management": "",
    "additional_care": [],
    "weekly_checklist": [],
    "important_warnings": []
}}

Rules:
- Consider the identified plant's actual needs.
- Do not recommend watering on a fixed schedule without considering soil moisture.
- Do not recommend fertilizer unless appropriate.
- Avoid unnecessary chemicals.
- Never invent environmental conditions.
- If information is missing, say what the owner should check.
- Make recommendations simple and practical.
- Return valid JSON only.
"""


# --- AI helpers --------------------------------------------------------------
@st.cache_resource
def get_client(api_key):
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )


def ask_model(client, prompt, image=None):
    """Send a prompt (and optional image) to the text model; retry temporary server errors."""
    contents = [prompt, image] if image is not None else prompt

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(model=TEXT_MODEL, contents=contents)
            return response.text.strip()
        except errors.ServerError:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(5 * attempt)


def parse_json_response(text):
    """Parse a JSON reply, tolerating Markdown code fences."""
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise ValueError("The AI did not return valid JSON. Please try again.") from error


def assess_severity(client, plant_data):
    prompt = f"""
Based on this plant health report, classify its visible condition.

Report:
{json.dumps(plant_data)}

Choose exactly one: {", ".join(SEVERITY_LEVELS)}.
Consider only the evidence in the report.
Return only the classification word.
"""
    answer = ask_model(client, prompt).strip(" .*\n").capitalize()
    return answer if answer in SEVERITY_LEVELS else "Uncertain"


def run_analysis(client, image):
    plant_data = parse_json_response(ask_model(client, ANALYSIS_PROMPT, image))
    severity = assess_severity(client, plant_data)
    care_plan = parse_json_response(
        ask_model(client, CARE_PLAN_PROMPT.format(report=json.dumps(plant_data, indent=2)))
    )
    return plant_data, severity, care_plan


# --- Recovery images (local Stable Diffusion) ---------------------------------
@st.cache_resource(show_spinner="Loading the image model (first run downloads ~4 GB)...")
def load_sd_pipeline():
    # Imported here so the app starts fast and works even without torch installed.
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        SD_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    ).to(device)
    return pipe


def generate_recovery_images(image, on_progress):
    import torch

    pipe = load_sd_pipeline()
    base_image = image.resize((512, 512))
    results = {}

    for index, (label, prompt, strength) in enumerate(RECOVERY_STAGES):
        on_progress(index / len(RECOVERY_STAGES), f"Generating: {label}...")
        if prompt is None:
            results[label] = base_image
            continue
        results[label] = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=base_image,
            strength=strength,
            guidance_scale=8,
            num_inference_steps=20,
            generator=torch.Generator("cpu").manual_seed(42),  # keeps the plant consistent
        ).images[0]

    on_progress(1.0, "Done!")
    return results


# --- UI helpers --------------------------------------------------------------
def show_list(items):
    if not items:
        st.caption("No information available.")
        return
    for item in items:
        st.markdown(f"- {item}")


def show_health_report(plant_data, severity):
    identity = plant_data.get("plant_identification", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("Plant", identity.get("common_name", "Unknown"))
    col2.metric("Scientific name", identity.get("scientific_name", "Unknown"))
    col3.metric("ID confidence", identity.get("identification_confidence", "Unknown"))

    st.markdown(f"### Overall condition: :{SEVERITY_COLORS[severity]}[**{severity}**]")

    st.subheader("🔍 Visible symptoms")
    show_list(plant_data.get("visible_symptoms"))

    st.subheader("⚠️ Possible health problems")
    problems = plant_data.get("possible_health_problems", [])
    if not problems:
        st.caption("No information available.")
    for problem in problems:
        with st.expander(f"{problem.get('problem', 'Unknown')} — likelihood: {problem.get('likelihood', '?')}"):
            st.write(problem.get("reason", ""))

    st.subheader("🌤️ Environmental factors")
    show_list(plant_data.get("environmental_factors"))

    st.subheader("💧 Recommended actions")
    show_list(plant_data.get("recommended_actions"))

    st.subheader("📅 Monitoring advice")
    show_list(plant_data.get("monitoring_advice"))

    st.subheader("📝 Overall assessment")
    st.write(plant_data.get("overall_assessment", "No assessment available."))


def show_care_plan(plan):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💧 Watering")
        watering = plan.get("watering", {})
        st.write(watering.get("recommendation", ""))
        st.caption(f"Frequency: {watering.get('frequency', 'n/a')}")
        if watering.get("warning"):
            st.warning(watering["warning"])

    with col2:
        st.subheader("☀️ Sunlight")
        sunlight = plan.get("sunlight", {})
        st.write(sunlight.get("recommendation", ""))
        st.caption(f"Ideal conditions: {sunlight.get('ideal_conditions', 'n/a')}")

    st.subheader("🪴 Soil and drainage")
    st.write(plan.get("soil_and_drainage", "No information available."))

    st.subheader("🐛 Pest management")
    st.write(plan.get("pest_management", "No information available."))

    st.subheader("🌿 Additional care")
    show_list(plan.get("additional_care"))

    st.subheader("✅ This week's checklist")
    for i, task in enumerate(plan.get("weekly_checklist", [])):
        st.checkbox(str(task), key=f"task_{i}")

    warnings = plan.get("important_warnings", [])
    if warnings:
        st.subheader("🚨 Important warnings")
        for warning in warnings:
            st.warning(str(warning))


def reset_results():
    for key in ("plant_data", "severity", "care_plan", "timeline"):
        st.session_state.pop(key, None)
    for key in [k for k in st.session_state if k.startswith("task_")]:
        del st.session_state[key]


# --- Page --------------------------------------------------------------------
st.set_page_config(page_title="Plant Doctor", page_icon="🌿", layout="wide")

st.title("🌿 Plant Doctor")
st.caption(
    "Upload a photo of your plant to get a health report, a care plan and a possible recovery timeline. "
    "⚠️ These are *possible* explanations from a single photo, not a definitive diagnosis."
)

with st.sidebar:
    st.header("Settings")
    api_key = os.environ.get("GEMINI_API_KEY") or st.text_input(
        "Google AI API key", type="password", help="Or set the GEMINI_API_KEY environment variable."
    )
    st.markdown("Get a key at [Google AI Studio](https://aistudio.google.com/apikey).")

if not api_key:
    st.info("👈 Enter your Google AI API key in the sidebar to start.")
    st.stop()

client = get_client(api_key)

uploaded = st.file_uploader("Upload a plant photo", type=["png", "jpg", "jpeg", "webp"])

if uploaded is None:
    st.stop()

file_id = f"{uploaded.name}-{uploaded.size}"
if st.session_state.get("file_id") != file_id:
    reset_results()
    st.session_state["file_id"] = file_id

plant_image = Image.open(uploaded).convert("RGB")
plant_image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

left, right = st.columns([1, 2])
left.image(plant_image, caption="Your plant", use_container_width=True)

with right:
    if st.button("🩺 Analyze my plant", type="primary"):
        try:
            with st.spinner("Examining your plant... this can take a minute."):
                plant_data, severity, care_plan = run_analysis(client, plant_image)
            st.session_state.update(plant_data=plant_data, severity=severity, care_plan=care_plan)
        except (errors.APIError, ValueError) as error:
            st.error(f"Something went wrong: {error}")

    if "plant_data" in st.session_state:
        st.success("Analysis complete! See the tabs below.")
        record = {
            "date": datetime.now().isoformat(),
            "severity": st.session_state["severity"],
            "plant_data": st.session_state["plant_data"],
            "care_plan": st.session_state["care_plan"],
        }
        st.download_button(
            "⬇️ Download report (JSON)",
            data=json.dumps(record, indent=2),
            file_name="plant_health_record.json",
            mime="application/json",
        )

if "plant_data" not in st.session_state:
    st.stop()

tab_report, tab_plan, tab_recovery = st.tabs(["🩺 Health report", "💧 Care plan", "📈 Recovery preview"])

with tab_report:
    show_health_report(st.session_state["plant_data"], st.session_state["severity"])

with tab_plan:
    show_care_plan(st.session_state["care_plan"])

with tab_recovery:
    st.caption(
        "AI-generated *possible* outcomes, not predictions. Runs on your own computer with Stable Diffusion: "
        "free, but slow on a CPU (a few minutes), and the first run downloads about 4 GB."
    )

    if st.button("🎨 Generate recovery preview"):
        progress = st.progress(0.0, text="Starting...")
        try:
            st.session_state["timeline"] = generate_recovery_images(
                plant_image, lambda value, text: progress.progress(value, text=text)
            )
        except ImportError:
            st.error("Missing packages. Run: pip install diffusers transformers accelerate torch")
        except Exception as error:  # download, memory or device problems
            st.error(f"Image generation failed: {error}")
        finally:
            progress.empty()

    timeline = st.session_state.get("timeline")
    if timeline:
        columns = st.columns(len(timeline))
        for column, (label, image) in zip(columns, timeline.items()):
            column.image(image, caption=label, use_container_width=True)
