import streamlit as st
import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.spatial.distance import cdist
from torchvision import transforms

vecs = np.load("all_vecs.npy")
names = np.load("all_names.npy")

model = torchvision.models.resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


model.avgpool.register_forward_hook(get_activation("avgpool"))


uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])



with torch.no_grad():
    if uploaded_file is not None:
        st.image(uploaded_file,width=300)
        uploaded_image = Image.open(uploaded_file)

        uploaded_image_tensor = transform(uploaded_image)
        out = model(uploaded_image_tensor[None, ...])
        uploaded_image_vec = activation["avgpool"].numpy().squeeze()[None, ...]



        top5_indices = cdist(uploaded_image_vec, vecs).squeeze().argsort()[1:6]


        if len(top5_indices) == 0:
            st.write('Aucune image similaire')
        else:

            st.write('Les images similaires :')
            c1, c2, c3, c4, c5 = st.columns(5)
            columns = [c1, c2, c3, c4, c5]
            for i, idx in enumerate(top5_indices):
                if i < 5:  # Assure que nous avons au maximum 5 images similaires
                    columns[i].image(Image.open(names[idx]))