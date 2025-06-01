from flask import Flask, redirect, render_template, request
from NeuralNet import NeuralNet
from PIL import Image
from torchvision import transforms
import torch
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diagnosis', methods=['POST'])
def diagnosis():
    # print(request.files)
    # print(request.files['file'].filename == "")
    if "file" not in request.files or request.files['file'].filename == '':
        return redirect("/")
    
    file = request.files['file']

    model = NeuralNet()
    model.load_state_dict(torch.load(os.path.join(os.curdir, "AI Model", "AI Model.pt"), weights_only=True, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])
        tensor = transform(Image.open(file))
        # print(tensor.shape)
        outputs = model(tensor[None, ...])

        _, predicted = torch.max(outputs.data, 1)

        print(predicted)

    return 'Hello, World'