import numpy as np
import gradio as gr


def sepia(input_img, request: gr.Request):
    print("Request headers dictionary:", request.headers)
    print("IP address:", request.client.host)
    print(f"{type(input_img)}, {input_img.shape}")
    sepia_filter = np.array(
        [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]
    )
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img


demo = gr.Interface(
    fn=sepia, inputs=gr.Image(height=200, width=200, type="numpy"), outputs="image"
)
demo.launch(share=True)
