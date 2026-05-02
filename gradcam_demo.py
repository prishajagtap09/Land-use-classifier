import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from prepare_data import NUM_CLASSES, CLASS_NAMES, MEAN, STD

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
model.eval().to(DEVICE)

# Hook to capture gradients from last conv layer
gradients, activations = [], []

def save_gradient(grad): gradients.append(grad)

target_layer = model.conv_head
target_layer.register_forward_hook(
    lambda m, i, o: (activations.append(o), o.register_hook(save_gradient))
)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

def gradcam(image_path):
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    out = model(tensor)
    pred_class = out.argmax().item()
    model.zero_grad()
    out[0, pred_class].backward()

    grad = gradients[-1].squeeze().mean(dim=[1, 2], keepdim=True)
    act  = activations[-1].squeeze()
    cam  = (grad * act).sum(0).relu()
    cam  = (cam - cam.min()) / (cam.max() + 1e-8)
    cam  = cam.detach().cpu().numpy()
    cam  = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((64, 64)))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img.resize((64, 64)))
    axes[0].set_title(f'Input image')
    axes[0].axis('off')
    axes[1].imshow(img.resize((64, 64)))
    axes[1].imshow(cam, alpha=0.5, cmap='jet')
    axes[1].set_title(f'Predicted: {CLASS_NAMES[pred_class]}')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig('gradcam_output.png', dpi=150)
    plt.show()

# Usage: gradcam('path/to/any/satellite_tile.jpg')