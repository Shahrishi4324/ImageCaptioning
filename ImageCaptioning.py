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

class DecoderWithAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim, encoder_dim, decoder_dim, dropout=0.5):
        super(DecoderWithAttention, self).__init__()
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(embed_size + encoder_dim, hidden_size, bias=True)
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        self.f_beta = nn.Linear(hidden_size, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, features, captions):
        batch_size = features.size(0)
        vocab_size = self.vocab_size
        caption_lengths = [len(caption) for caption in captions]

        embeddings = self.embedding(captions)

        h, c = self.init_hidden_state(features)

        decode_lengths = [length - 1 for length in caption_lengths]

        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(features.device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(features[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1), (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds

        return predictions

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

# Initialize the decoder
decoder = DecoderWithAttention(embed_size=256, hidden_size=512, vocab_size=5000, attention_dim=256, encoder_dim=2048, decoder_dim=512)

import torch.optim as optim

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(list(decoder.parameters()) + list(encoder.parameters()), lr=1e-4)

# Training loop
for epoch in range(10):
    encoder.train()
    decoder.train()
    epoch_loss = 0
    
    for i, (images, captions) in enumerate(dataloader):
        images = images.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()
        
        features = encoder(images)
        outputs = decoder(features, captions)

        targets = captions[:, 1:]  # Ignore the <start> token
        loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.contiguous().view(-1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if i % 10 == 0:
            print(f"Epoch {epoch + 1}, Step {i}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader):.4f}")

import matplotlib.pyplot as plt
import numpy as np

def generate_caption(image, encoder, decoder, vocab, max_len=20):
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        features = encoder(image.unsqueeze(0))
        h, c = decoder.init_hidden_state(features)
        
        caption = []
        alpha_list = []
        
        for _ in range(max_len):
            attention_weighted_encoding, alpha = decoder.attention(features, h)
            h, c = decoder.decode_step(attention_weighted_encoding, (h, c))
            preds = decoder.fc(h)
            word_id = preds.argmax(1).item()
            word = vocab[word_id]
            caption.append(word)
            alpha_list.append(alpha.cpu().numpy())
            
            if word == "<end>":
                break
        
        return caption, alpha_list

# Example usage
sample_image = images[0].to(device)
caption, attention_weights = generate_caption(sample_image, encoder, decoder, vocab={0: '<start>', 1: 'a', 2: 'cat', 3: '<end>'})

# Visualize the attention maps
def visualize_attention(image, caption, attention_weights):
    fig = plt.figure(figsize=(15, 15))
    for idx, word in enumerate(caption):
        if idx == len(attention_weights):
            break
        ax = fig.add_subplot(len(caption) // 5 + 1, 5, idx + 1)
        ax.set_title(word)
        attention_map = attention_weights[idx].reshape(7, 7)
        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax.imshow(attention_map, alpha=0.6, extent=(0, 224, 224, 0), interpolation='bilinear', cmap='gray')
        plt.axis('off')

# Visualizing the attention for a generated caption
visualize_attention(sample_image, caption, attention_weights)
plt.show()