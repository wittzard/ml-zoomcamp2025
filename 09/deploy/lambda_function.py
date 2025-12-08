import json
import numpy as np
from PIL import Image
import onnxruntime as ort
import io
import base64

session = ort.InferenceSession("hair_classifier_empty.onnx")
input_name = session.get_inputs()[0].name


from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def lambda_handler(event, context):
    image_data = event["image"]
    image_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img_tensor = preprocess(img).unsqueeze(0).numpy().astype(np.float32)

    output = session.run(None, {input_name: img_tensor})

    # Return scalar output (model predicts a single value)
    return {"prediction": float(output[0][0][0])}
