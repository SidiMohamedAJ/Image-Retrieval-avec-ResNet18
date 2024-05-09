import os

import torchvision
from PIL import Image
import numpy as np
import torch
from torchvision import transforms



def extract_features(root):
    images = []

    # Parcourir tous les dossiers et fichiers dans root
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            img_path = os.path.join(dirpath, filename)
            images.append(img_path)

    model = torchvision.models.resnet18(pretrained=True)

    all_vecs = None
    all_names = []

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    model.avgpool.register_forward_hook(get_activation("avgpool"))

    with torch.no_grad():
        for i, image_path in enumerate(images):
            try:
                img = Image.open(image_path)
                img = transform(img)
                out = model(img[None, ...])
                vec = activation["avgpool"].numpy().squeeze()[None, ...]
                if all_vecs is None:
                    all_vecs = vec
                else:
                    all_vecs = np.vstack([all_vecs, vec])
                all_names.append(image_path)
            except:
                continue

    return all_vecs, all_names, model

def main():
    root = r"database\train"
    all_vecs, all_names, model = extract_features(root)
    np.save("all_vecs.npy", all_vecs)
    np.save("all_names.npy", all_names)


if __name__ == "__main__":
    main()
