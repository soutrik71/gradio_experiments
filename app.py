from transformers import pipeline
import gradio as gr

# Load the summarization model once
model = pipeline("summarization")


# Prediction function
def predict(prompt):
    try:
        # Generate summary and return
        summary = model(prompt, max_length=150, min_length=30, do_sample=False)[0][
            "summary_text"
        ]
        return summary
    except Exception as e:
        return f"Error: {str(e)}"


# Gradio interface
with gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Enter text to summarize", placeholder="Type your content here..."
    ),
    outputs=gr.Textbox(label="Summary"),
    title="Text Summarizer",
    description="Enter text and get a concise summary powered by Hugging Face transformers.",
) as interface:
    interface.launch()
