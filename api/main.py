import PIL.Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from torchvision import transforms
import torch
import os
import io
from huggingface_hub import hf_hub_download
from model import SiameseNetwork

app = FastAPI()

@app.get('/')
def hello_app():
    return {"Hello": "Enter your input face, This Siamese model is going to estimate the similarity degree between them"}

# Download the model from Hugging Face Hub in the working directory
cwd = os.getcwd()
model_path = hf_hub_download(repo_id="MohamedOussama/FaceSimNet", filename="siamese_model.pth", revision="main", cache_dir=cwd)
# Load the state dictionary
siamese_model = SiameseNetwork()
directory = os.path.join('models--MohamedOussama--FaceSimNet', 'snapshots', 'b51e3d309d26931109a4b77bfcde2cbff586f7e2')
siamese_model.load_state_dict(torch.load(os.path.join(directory, "siamese_model.pth"), map_location=torch.device('cpu')))
# Move the model to the desired device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
siamese_model.to(device)



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to configure inputs so that can be accepted by the model
def load_image(content):
    img = PIL.Image.open(io.BytesIO(content))
    img = img.convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    return img.to(device)

@app.post('/predict')
async def similarity_estimate(input_1: UploadFile = File(...), input_2: UploadFile = File(...)):
    content_1 = await input_1.read()
    content_2 = await input_2.read()

    img_1 = load_image(content_1)
    img_2 = load_image(content_2)
    
    with torch.no_grad():
        output = siamese_model(img_1, img_2)
         
    similarity = output.item()    

    return JSONResponse(content={"similarity": similarity})
    