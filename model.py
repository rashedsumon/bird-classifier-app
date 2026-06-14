# model.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def get_transform_pipeline():
    """
    Returns standard evaluation transforms aligned with ImageNet weights.
    The dataset images are 224x224.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

def initialize_model(num_classes=20):
    """
    Constructs a lightweight MobileNetV2 architecture with custom output heads.
    """
    # Using lightweight MobileNetV2 for fast inferences on Streamlit CPU cloud instances
    model = models.mobilenet_v2(pretrained=True)
    
    # Mutate the classifier head to output predictions for 20 classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    
    # Set to evaluation mode for inference stability
    model.eval()
    return model

def predict_bird_species(image_input, model, class_names):
    """
    Process raw image input and evaluates probabilities using softmax.
    """
    transform = get_transform_pipeline()
    
    # Convert image to RGB matrix (drops alpha layer if present)
    img = Image.open(image_input).convert("RGB")
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)

    with torch.no_grad():
        outputs = model(batch_t)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Grab top performance match
    confidence, index = torch.max(probabilities, dim=0)
    
    return class_names[index.item()], confidence.item()