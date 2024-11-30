import gradio as gr


def test(name, checkbox, value):
    return f"{name=}, {checkbox=}, {value=}"


demo = gr.Interface(
    fn=test,
    inputs=[gr.Text(), gr.Checkbox(), gr.Slider(0, 100)],
    outputs=gr.Text(),
    title="Multi Inputs",
    description="A simple example with multiple inputs",
)

demo.launch(share=True)
