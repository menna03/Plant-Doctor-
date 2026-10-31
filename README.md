# Plant-Doctor-

Plant Doctor is an AI plant-care assistant. Upload a photo of a sick or struggling plant and get a clear health report, a practical care plan, and a visual preview of how the plant could recover.

What it does

Reads the photo: Gemma 4 (multimodal, through the Gemini API) identifies the plant, lists only the symptoms visible in the image, and suggests possible causes with likelihoods. It never claims a definitive diagnosis and it explains its uncertainty.

Rates severity: a second step classifies the plant as Healthy, Mild, Moderate, Severe or Uncertain and shows it as a color-coded label.

Builds a care plan: watering, sunlight, soil and drainage, pest management, warnings, and an interactive weekly checklist.

Shows a recovery timeline: Day 1, Week 1, Week 2 and a final image, generated locally with open-source Stable Diffusion (img2img) so the plant in the pictures stays recognizable.

Exports the report as a JSON file so users can compare later check-ups.

How it works

The app asks Gemma for structured JSON, checks it, and shows it in a Streamlit interface with tabs for the health report, the care plan and the recovery preview. It retries temporary server errors and shows readable error messages. The recovery images run on the user’s own computer, so that feature needs no paid API.

Why we built it

Many people lose houseplants because the symptoms of overwatering, underwatering and low humidity look alike. Plant Doctor gives beginners clear, cautious guidance instead of a confident but possibly wrong diagnosis.

Limitations

Results are possible explanations from one photo, not a diagnosis. The recovery images are illustrative, not predictions. Local image generation is slow on a CPU, and the first run downloads about 4 GB.


Demo: https://drive.google.com/file/d/1uIwHqUgrSsIZxOKoWME6NGfTRoHT_y02/view?usp=drive_link
