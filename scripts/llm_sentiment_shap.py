import shap
import transformers
from transformers import pipeline

# 1. Load Model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
p = pipeline("text-classification", model=model_name, return_all_scores=True)

# 2. Define Input
text_inputs = [
    "I really wanted to hate this movie because the actor is annoying, but the plot was actually fantastic."
]
print("Processing text with SHAP... (this may take a moment)")

# 3. Compute SHAP
explainer = shap.Explainer(p)
shap_values = explainer(text_inputs)

# 4. Visualize
html_visualization = shap.plots.text(shap_values, display=False)

output_filename = "../outputs/shap_llm_output.html"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(html_visualization)

print(f"\nSuccess! Open '{output_filename}' in your browser.")
