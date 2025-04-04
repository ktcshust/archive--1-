import os
import csv
import time
import nltk
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from bert_score import score as bert_score_score
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Tải nltk packages nếu cần
nltk.download('punkt')

############################################
# 1. Dataset: đọc file CSV với 2 cột "image", "caption"
############################################
class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None):
        """
        Args:
            root_dir: đường dẫn folder chứa ảnh.
            captions_file: file CSV với 2 cột "image", "caption"
            transform: các transform cho ảnh.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = []
        self.captions = []
        with open(captions_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.imgs.append(row["image"])
                self.captions.append(row["caption"])
    def __len__(self):
        return len(self.captions)
    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img_path = os.path.join(self.root_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, caption  # caption là string

def collate_fn(data):
    images, captions = zip(*data)
    images = torch.stack(images, 0)
    return images, list(captions)

############################################
# 2. Module GCN cho visual features
############################################
def build_grid_adj(H, W):
    num_nodes = H * W
    A = torch.zeros(num_nodes, num_nodes)
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            A[idx, idx] = 1.0  # self-loop
            if i > 0:
                A[idx, (i-1)*W + j] = 1.0
            if i < H - 1:
                A[idx, (i+1)*W + j] = 1.0
            if j > 0:
                A[idx, i*W + (j-1)] = 1.0
            if j < W - 1:
                A[idx, i*W + (j+1)] = 1.0
    D = A.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
    def forward(self, x, A_norm):
        # x: (batch, num_nodes, in_dim)
        out = torch.bmm(A_norm.unsqueeze(0).expand(x.size(0), -1, -1), x)
        out = self.linear(out)
        out = self.relu(out)
        return out

############################################
# 3. Encoder: CNN -> GCN -> Full-Memory Transformer -> Skip-Connections
############################################
class FullMemoryTransformer(nn.Module):
    def __init__(self, encoder_dim, nhead=8, num_layers=2, dim_feedforward=2048, dropout=0.1):
        super(FullMemoryTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=encoder_dim, nhead=nhead,
                                                    dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, features):
        # features: (batch, num_pixels, encoder_dim)
        features = features.permute(1, 0, 2)  # (num_pixels, batch, encoder_dim)
        transformed = self.transformer(features)
        transformed = transformed.permute(1, 0, 2)
        return transformed

class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(EncoderCNN, self).__init__()
        self.enc_image_size = encoded_image_size
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]  # loại bỏ fc và avgpool
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        # GCN layer
        self.gcn = GCNLayer(2048, 2048)
        A_norm = build_grid_adj(encoded_image_size, encoded_image_size)
        self.register_buffer("A_norm", A_norm)
        self.full_memory_transformer = FullMemoryTransformer(encoder_dim=2048, nhead=8, num_layers=2)
    def forward(self, images):
        features = self.resnet(images)  # (batch, 2048, feat_size, feat_size)
        features = self.adaptive_pool(features)  # (batch, 2048, enc_image_size, enc_image_size)
        features = features.permute(0, 2, 3, 1)  # (batch, H, W, 2048)
        batch_size, H, W, encoder_dim = features.size()
        features_flat = features.view(batch_size, H * W, encoder_dim)
        gcn_features = self.gcn(features_flat, self.A_norm)
        enriched = features_flat + gcn_features
        transformer_features = self.full_memory_transformer(enriched)
        combined = transformer_features + enriched  # Skip-connections
        return combined  # (batch, num_pixels, 2048)

############################################
# 4. Hash Memory Module để áp dụng lên visual embeddings
############################################
class HashMemory(nn.Module):
    def __init__(self, hidden_dim, memory_size=128):
        super(HashMemory, self).__init__()
        self.memory_bank = nn.Parameter(torch.randn(memory_size, hidden_dim))
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        similarity = torch.matmul(x, self.memory_bank.t())  # (batch, seq_len, memory_size)
        att_weights = torch.softmax(similarity, dim=-1)
        memory_out = torch.matmul(att_weights, self.memory_bank)  # (batch, seq_len, hidden_dim)
        return x + memory_out

############################################
# 5. Visual Captioning Model: kết hợp visual encoder với T5 decoder, tích hợp hash memory
############################################
class VisualCaptioningModel(nn.Module):
    def __init__(self, d_model=512, encoded_image_size=14):
        super(VisualCaptioningModel, self).__init__()
        self.visual_encoder = EncoderCNN(encoded_image_size)
        self.proj = nn.Linear(2048, d_model)  # chuyển 2048-d sang 512-d cho T5
        # Tích hợp Hash Memory cho visual embeddings
        self.hash_memory = HashMemory(d_model, memory_size=128)
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-base')
    def forward(self, images, labels_input_ids, labels_attention_mask):
        encoder_out = self.visual_encoder(images)  # (batch, num_pixels, 2048)
        proj_features = self.proj(encoder_out)       # (batch, num_pixels, 512)
        proj_features = self.hash_memory(proj_features)  # tích hợp hash memory
        encoder_attention_mask = torch.ones(proj_features.size()[:2], dtype=torch.long, device=proj_features.device)
        encoder_outputs = self.t5.encoder(inputs_embeds=proj_features,
                                          attention_mask=encoder_attention_mask)
        outputs = self.t5(decoder_input_ids=labels_input_ids,
                          attention_mask=labels_attention_mask,
                          encoder_outputs=(encoder_outputs,),
                          labels=labels_input_ids)
        return outputs
    def generate_caption(self, images, max_length=20):
        encoder_out = self.visual_encoder(images)  # (batch, num_pixels, 2048)
        proj_features = self.proj(encoder_out)       # (batch, num_pixels, 512)
        proj_features = self.hash_memory(proj_features)
        encoder_attention_mask = torch.ones(proj_features.size()[:2], dtype=torch.long, device=proj_features.device)
        encoder_outputs = self.t5.encoder(inputs_embeds=proj_features,
                                          attention_mask=encoder_attention_mask)
        generated_ids = self.t5.generate(encoder_outputs=(encoder_outputs,), max_length=max_length)
        return generated_ids

############################################
# 6. Evaluation Metrics functions
############################################
def compute_bleu(refs, hyps):
    smoothie = SmoothingFunction().method4
    bleu_scores = {1: [], 2: [], 3: [], 4: []}
    for ref, hyp in zip(refs, hyps):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        bleu_scores[1].append(sentence_bleu([ref_tokens], hyp_tokens, weights=(1,0,0,0), smoothing_function=smoothie))
        bleu_scores[2].append(sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5,0.5,0,0), smoothing_function=smoothie))
        bleu_scores[3].append(sentence_bleu([ref_tokens], hyp_tokens, weights=(0.33,0.33,0.33,0), smoothing_function=smoothie))
        bleu_scores[4].append(sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoothie))
    return np.mean(bleu_scores[1]), np.mean(bleu_scores[2]), np.mean(bleu_scores[3]), np.mean(bleu_scores[4])

def compute_cider(refs, hyps):
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score({str(i): [r] for i, r in enumerate(refs)},
                                          {str(i): [h] for i, h in enumerate(hyps)})
    return score

def compute_rouge(refs, hyps):
    rouge_scorer = Rouge()
    score, _ = rouge_scorer.compute_score({str(i): [r] for i, r in enumerate(refs)},
                                          {str(i): [h] for i, h in enumerate(hyps)})
    return score

def compute_spice(refs, hyps):
    spice_scorer = Spice()
    score, _ = spice_scorer.compute_score({str(i): [r] for i, r in enumerate(refs)},
                                          {str(i): [h] for i, h in enumerate(hyps)})
    return score

def compute_bertscore(refs, hyps, lang="en"):
    P, R, F1 = bert_score_score(hyps, refs, lang=lang, verbose=False)
    return torch.mean(F1).item()

############################################
# 7. Evaluation: Tính các metric và Avg Inference Time
############################################
def evaluate_model(model, dataloader, tokenizer, device):
    model.eval()
    refs = []
    hyps = []
    total_time = 0.0
    count = 0
    with torch.no_grad():
        for imgs, captions in dataloader:
            imgs = imgs.to(device)
            tokenized = tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
            labels_input_ids = tokenized.input_ids.to(device)
            labels_attention_mask = tokenized.attention_mask.to(device)
            for i in range(imgs.size(0)):
                start_time = time.time()
                gen_ids = model.generate_caption(imgs[i].unsqueeze(0), max_length=20)
                total_time += (time.time() - start_time)
                count += 1
                generated = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                hyps.append(generated)
            refs.extend(captions)
    avg_time = total_time / count if count > 0 else 0.0
    bleu1, bleu2, bleu3, bleu4 = compute_bleu(refs, hyps)
    cider = compute_cider(refs, hyps)
    rouge = compute_rouge(refs, hyps)
    spice = compute_spice(refs, hyps)
    bertscore = compute_bertscore(refs, hyps, lang="en")
    return bleu1, bleu2, bleu3, bleu4, cider, rouge, spice, bertscore, avg_time

############################################
# 8. Main Training & Evaluation
############################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = "archive (3)/Images"       # đường dẫn folder ảnh
    captions_file = "archive (3)/captions.txt"  # file CSV với 2 cột: image, caption
    num_epochs = 25
    batch_size = 32
    learning_rate = 1e-4
    encoded_image_size = 14
    d_model = 512  # hidden size của T5-base

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])

    dataset = Flickr8kDataset(root_dir, captions_file, transform=transform)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = VisualCaptioningModel(d_model=d_model, encoded_image_size=encoded_image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop: sử dụng T5's training objective
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for imgs, captions in train_loader:
            imgs = imgs.to(device)
            tokenized = tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
            labels_input_ids = tokenized.input_ids.to(device)
            labels_attention_mask = tokenized.attention_mask.to(device)
            optimizer.zero_grad()
            outputs = model(images=imgs, labels_input_ids=labels_input_ids, labels_attention_mask=labels_attention_mask)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluation
    bleu1, bleu2, bleu3, bleu4, cider, rouge, spice, bertscore, avg_inference_time = evaluate_model(model, test_loader, tokenizer, device)
    print("Evaluation Metrics on Test Set:")
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")
    print(f"CIDEr: {cider:.4f}")
    print(f"ROUGE-L: {rouge:.4f}")
    print(f"SPICE: {spice:.4f}")
    print(f"BERTScore F1: {bertscore:.4f}")
    print(f"Avg Inference Time per Image: {avg_inference_time:.4f} seconds")
    
    # Lưu model
    torch.save(model.state_dict(), "visual_captioning_t5_hash.pth")

if __name__ == '__main__':
    main()

