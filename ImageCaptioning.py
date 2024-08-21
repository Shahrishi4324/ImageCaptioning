import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import json
import nltk

# Download necessary NLTK data
nltk.download('punkt')

# Path configurations
image_folder = "path/to/coco/images"
caption_file = "path/to/coco/captions.json"

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load and preprocess captions
with open(caption_file, 'r') as f:
    captions_data = json.load(f)

# Tokenization of captions
def tokenize_caption(caption):
    return nltk.word_tokenize(caption.lower())

class CocoDataset(Dataset):
    def __init__(self, image_folder, captions_data, transform):
        self.image_folder = image_folder
        self.captions_data = captions_data
        self.transform = transform
        self.image_ids = list(captions_data.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        caption = self.captions_data[image_id][0]  # Using the first caption for simplicity
        image_path = os.path.join(self.image_folder, image_id + '.jpg')
        
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        caption_tokens = tokenize_caption(caption)
        
        return image, caption_tokens

# Initialize dataset and dataloader
dataset = CocoDataset(image_folder, captions_data, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Sample a batch
images, captions = next(iter(dataloader))
print(images.shape, len(captions))

import torch.nn as nn
import torchvision.models as models

# Load a pre-trained ResNet50 model
resnet = models.resnet50(pretrained=True)

# Remove the last fully connected layer
resnet = nn.Sequential(*list(resnet.children())[:-1])

# Freeze the parameters
for param in resnet.parameters():
    param.requires_grad = False

# Example of extracting features
sample_image = images[0].unsqueeze(0)  # Taking the first image from the batch
with torch.no_grad():
    feature_vector = resnet(sample_image)

print(feature_vector.shape)  # Should be [1, 2048, 1, 1] or similar depending on architecture

class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        self.resnet = resnet
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = self.relu(features)
        features = self.dropout(features)
        return features

# Initialize the encoder
encoder = Encoder(embed_size=256)
sample_features = encoder(sample_image)
print(sample_features.shape)  # Should be [1, 256]

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  
        self.full_att = nn.Linear(attention_dim, 1)  
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return attention_weighted_encoding, alpha