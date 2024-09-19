import torch
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_path, max_size=400, shape=None):
    image = Image.open(image_path).convert('RGB')
    
    if shape is not None:
        size = shape 
        in_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])
    else:
        if max(image.size) > max_size:
            size = max_size
        else:
            size = max(image.size)
        
        # Resize maintaining aspect ratio
        in_transform = transforms.Compose([
            transforms.Resize((size, int(size * image.size[1] / image.size[0]))),
            transforms.ToTensor(),
        ])

    # Discard the transparent, alpha channel (if it exists) and add the batch dimension
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image

# Function to convert a tensor to an image
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image

# Load images
content_image = load_image('/Users/krishnaarora/Desktop/PRO/5/content.jpeg').to('cpu')
style_image = load_image('/Users/krishnaarora/Desktop/PRO/5/style.jpeg', shape=content_image.shape[-2:]).to('cpu')

vgg = models.vgg19(weights='DEFAULT').features

for param in vgg.parameters():
    param.requires_grad_(False)

content_layers = ['28'] 
style_layers = ['0', '5', '10', '19', '28']

# Function to extract features
def get_features(image, model):
    features = {}
    x = image
    for name, layer in enumerate(model):
        x = layer(x)
        if str(name) in content_layers:
            features['content'] = x
        elif str(name) in style_layers:
            features[name] = x
    return features

# Define the loss function
def get_loss(content_features, style_features, target_features, content_weight=1e4, style_weight=1e2):
    content_loss = torch.mean((target_features['content'] - content_features['content']) ** 2)
    style_loss = 0
    for layer in style_layers:
        if layer in target_features:
            _, d, h, w = target_features[layer].shape
            target_gram = torch.matmul(target_features[layer].view(d, -1), target_features[layer].view(d, -1).t())
            style_gram = torch.matmul(style_features[layer].view(d, -1), style_features[layer].view(d, -1).t())
            style_loss += torch.mean((target_gram - style_gram) ** 2) / (d * h * w)
    
    total_loss = content_weight * content_loss + style_weight * style_loss
    return total_loss

# Style Transfer Function
def style_transfer(content_image, style_image, model, iterations=500, lr=0.01):
    # Extract features
    content_features = get_features(content_image, model)
    style_features = get_features(style_image, model)
    
    target_image = content_image.clone().requires_grad_(True)
    optimizer = optim.Adam([target_image], lr=lr)
    
    for i in range(1, iterations + 1):
        target_features = get_features(target_image, model)
        loss = get_loss(content_features, style_features, target_features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Iteration {i}: Loss = {loss.item()}")
    
    return target_image

# Perform style transfer
result_image = style_transfer(content_image, style_image, vgg)

# Display the result
plt.imshow(im_convert(result_image))
plt.axis('off')
plt.show()
