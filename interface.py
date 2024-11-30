import gradio as gr


def greetings(name, is_morning, temperature):

    salutation = "Good morning" if is_morning else "Good evening"
    greeting = f"{salutation} {name}. It is {temperature} degrees today"
    celsius = (temperature - 32) * 5 / 9
    return greeting, round(celsius, 2)


demo = gr.Interface(
    fn=greetings,
    inputs=[
        gr.Textbox(label="Name"),
        gr.Checkbox(label="Is Morning?"),
        gr.Slider(0, 100, label="Temperature"),
    ],
    outputs=[
        gr.Textbox(label="Greetings", lines=1),
        gr.Number(label="Temperature in Celsius"),
    ],
)

demo.launch(share=True)
