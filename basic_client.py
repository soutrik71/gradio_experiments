from gradio_client import Client

client = Client("http://127.0.0.1:7860/")
job = client.submit(num1=3, operation="add", num2=3, api_name="/predict")
print(job.result())
