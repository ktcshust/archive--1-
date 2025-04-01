import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import json
from collections import Counter
import numpy as np
import random
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from pycocoevalcap.cider.cider import Cider
import re

# Tải xuống các tài nguyên cần thiết của NLTK (chỉ cần chạy lần đầu tiên)
nltk.download('punkt')

# --------------------------------------------------------------------------------
# 1. Cấu hình
# --------------------------------------------------------------------------------

# Đường dẫn đến dữ liệu Flickr8k
DATA_DIR = './Flickr8k'
IMAGE_DIR = os.path.join(DATA_DIR, 'Flicker8k_Dataset')
CAPTION_FILE = os.path.join(DATA_DIR, 'Flickr8k_text/Flickr8k.token.txt')

# Các tham số huấn luyện
BATCH_SIZE = 32
EMBED_SIZE = 256  # Kích thước vector embedding từ
HIDDEN_SIZE = 512  # Kích thước trạng thái ẩn của LSTM
ATTENTION_SIZE = 512  # Kích thước của lớp Attention
NUM_EPOCHS = 20    # Số lượng epochs huấn luyện
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5
CLIP_GRAD = 5.0  # Giá trị để clip gradient
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Kích thước của Expansion Block
EXPANSION_SIZE = 2048

# Tỉ lệ train/test
TRAIN_RATIO = 0.8

# --------------------------------------------------------------------------------
# 2. Tiền xử lý dữ liệu
# --------------------------------------------------------------------------------

def load_captions(filename):
    """
    Tải chú thích từ file.

    Args:
        filename (str): Đường dẫn đến file chứa chú thích.

    Returns:
        dict: Một dictionary với key là tên ảnh và value là list các chú thích.
    """
    captions = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue  # Bỏ qua các dòng không hợp lệ
            image_name, caption = parts
            image_id, caption_id = image_name.split('#')
            if image_id not in captions:
                captions[image_id] = []
            captions[image_id].append(caption)
    return captions

def preprocess_captions(captions):
    """
    Tiền xử lý chú thích: chuyển về chữ thường, loại bỏ ký tự đặc biệt, thêm token start và end.

    Args:
        captions (dict): Dictionary chứa chú thích thô.

    Returns:
        dict: Dictionary chứa chú thích đã tiền xử lý.
    """
    processed_captions = {}
    for image_id, caption_list in captions.items():
        processed_captions[image_id] = []
        for caption in caption_list:
            caption = caption.lower()
            caption = re.sub(r'[^\w\s]', '', caption)  # Loại bỏ ký tự đặc biệt
            caption = '<start> ' + caption + ' <end>'
            processed_captions[image_id].append(caption)
    return processed_captions

def build_vocabulary(captions, vocab_threshold=4):
    """
    Xây dựng từ vựng từ các chú thích.

    Args:
        captions (dict): Dictionary chứa chú thích đã tiền xử lý.
        vocab_threshold (int): Ngưỡng tần suất xuất hiện của từ để được đưa vào từ vựng.

    Returns:
        dict: Một dictionary ánh xạ từ tới chỉ số.
    """
    word_counts = Counter()
    for caption_list in captions.values():
        for caption in caption_list:
            tokens = nltk.word_tokenize(caption)
            word_counts.update(tokens)

    vocab = {
        '<pad>': 0,
        '<start>': 1,
        '<end>': 2,
        '<unk>': 3  # Token cho các từ không có trong từ vựng
    }
    index = len(vocab)
    for word, count in word_counts.items():
        if count >= vocab_threshold:
            vocab[word] = index
            index += 1
    return vocab

def split_data(captions, train_ratio):
    """
    Chia dữ liệu thành tập train và test.

    Args:
        captions (dict): Dictionary chứa chú thích đã tiền xử lý.
        train_ratio (float): Tỉ lệ dữ liệu dành cho tập train.

    Returns:
        tuple: Hai dictionary, một cho tập train và một cho tập test.
    """
    image_ids = list(captions.keys())
    random.shuffle(image_ids)
    train_size = int(len(image_ids) * train_ratio)
    train_ids = image_ids[:train_size]
    test_ids = image_ids[train_size:]

    train_data = {img_id: captions[img_id] for img_id in train_ids}
    test_data = {img_id: captions[img_id] for img_id in test_ids}
    return train_data, test_data

class Flickr8kDataset(Dataset):
    """
    Dataset cho dữ liệu Flickr8k.
    """
    def __init__(self, image_dir, captions, vocab, transform=None):
        """
        Args:
            image_dir (str): Đường dẫn đến thư mục chứa ảnh.
            captions (dict): Dictionary chứa chú thích đã tiền xử lý.
            vocab (dict): Dictionary ánh xạ từ tới chỉ số.
            transform (callable, optional): Các biến đổi ảnh.
        """
        self.image_dir = image_dir
        self.captions = captions
        self.vocab = vocab
        self.transform = transform
        self.image_ids = list(captions.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Lấy một mẫu dữ liệu.

        Returns:
            tuple: (ảnh, chú thích đã mã hóa).
        """
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Chọn ngẫu nhiên một chú thích
        caption = random.choice(self.captions[image_id])
        tokens = nltk.word_tokenize(caption)
        encoded_caption = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        encoded_caption = torch.tensor(encoded_caption)
        return image, encoded_caption

def collate_fn(batch):
    """
    Hàm collate để xử lý các chú thích có độ dài khác nhau trong một batch.

    Args:
        batch (list): List các tuple (ảnh, chú thích).

    Returns:
        tuple: (ảnh đã stack, chú thích đã padding, độ dài của các chú thích).
    """
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)

    # Padding chú thích
    caption_lengths = [len(cap) for cap in captions]
    max_length = max(caption_lengths)
    padded_captions = torch.zeros(len(captions), max_length, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded_captions[i, :len(cap)] = cap

    return images, padded_captions, caption_lengths

# --------------------------------------------------------------------------------
# 3. Định nghĩa mô hình
# --------------------------------------------------------------------------------

class EncoderCNN(nn.Module):
    """
    Encoder CNN sử dụng ResNet-101 đã được tiền huấn luyện.
    """
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(pretrained=True)
        # Loại bỏ lớp fully connected cuối cùng
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        # Thay đổi kích thước output của feature map
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

    def forward(self, images):
        """
        Forward pass của encoder.

        Args:
            images (torch.Tensor): Tensor chứa ảnh đầu vào.

        Returns:
            torch.Tensor: Feature map của ảnh.
        """
        features = self.features(images)
        features = self.adaptive_pool(features)
        return features

class ExpansionBlock(nn.Module):
    """
    Expansion Block để tăng chiều sâu của feature map.
    """
    def __init__(self, in_channels, expansion_size):
        super(ExpansionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, expansion_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expansion_size)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(expansion_size, expansion_size, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(expansion_size)
        self.conv3 = nn.Conv2d(expansion_size, in_channels, kernel_size=1, bias=False) # Dự đoán lại số channels ban đầu
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Forward pass của Expansion Block.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Output feature map.
        """
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)
        out += identity  # Skip connection
        out = self.relu(out) # ReLU sau khi cộng
        return out

class Attention(nn.Module):
    """
    Lớp Attention.
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward pass của lớp Attention.

        Args:
            encoder_out (torch.Tensor): Output của encoder CNN.
            decoder_hidden (torch.Tensor): Trạng thái ẩn của decoder LSTM.

        Returns:
            tuple: (attention weights, context vector).
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, L, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att2 = att2.unsqueeze(1)  # (batch_size, 1, attention_dim)
        att = self.relu(att1 + att2)  # (batch_size, L, attention_dim)
        e = self.full_att(att).squeeze(2)  # (batch_size, L)
        alpha = self.softmax(e)  # (batch_size, L)

        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return alpha, context

class DecoderRNN(nn.Module):
    """
    Decoder RNN (LSTM) với Attention.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, attention_size, dropout_rate):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim=2048, decoder_dim=hidden_size, attention_dim=attention_size) # Thay đổi encoder_dim thành 2048
        self.lstm = nn.LSTMCell(embed_size + 2048, hidden_size) # Thay đổi input_size của LSTMCell
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def forward(self, features, captions, caption_lengths, hidden=None):
        """
        Forward pass của decoder.

        Args:
            features (torch.Tensor): Output của encoder CNN.
            captions (torch.Tensor): Tensor chứa các chú thích đã mã hóa.
            caption_lengths (list): List chứa độ dài của các chú thích.
            hidden (tuple, optional): Trạng thái ẩn ban đầu.

        Returns:
            tuple: (logits, attention weights).
        """
        batch_size = features.size(0)
        if hidden is None:
            hidden = (torch.zeros(batch_size, self.hidden_size).to(DEVICE),
                      torch.zeros(batch_size, self.hidden_size).to(DEVICE))

        # Sắp xếp features theo độ dài caption giảm dần (để sử dụng packed sequence nếu cần)
        sorted_cap_lengths, sorted_cap_indices = torch.sort(torch.tensor(caption_lengths, device=DEVICE), descending=True)
        features = features[sorted_cap_indices]
        captions = captions[sorted_cap_indices]

        # Embedding
        embeddings = self.embed(captions)  # (batch_size, max_length, embed_size)

        # Chuẩn bị để lưu kết quả
        logits = torch.zeros(batch_size, captions.size(1), self.vocab_size).to(DEVICE)
        att_weights = torch.zeros(batch_size, captions.size(1), features.size(1)).to(DEVICE) # Lưu attention weights

        # Initialize input to the LSTM: [context vector, embedding]
        for t in range(captions.size(1)):
            context, att_weight = self.attention(features, hidden[0])
            input_combined = torch.cat([context, embeddings[:, t]], dim=1)
            hidden = self.lstm(input_combined, hidden)
            output = self.dropout(hidden[0])
            logit = self.fc(output)
            logits[:, t] = logit
            att_weights[:, t] = att_weight # Lưu attention weights

        # Unsort logits and attention weights to the original order
        _, unsorted_indices = torch.sort(sorted_cap_indices)
        logits = logits[unsorted_indices]
        att_weights = att_weights[unsorted_indices]

        return logits, att_weights

class ImageCaptioningModel(nn.Module):
    """
    Mô hình Image Captioning tổng hợp.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, attention_size, dropout_rate, expansion_size): # Thêm expansion_size
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN()
        self.expansion_block = ExpansionBlock(in_channels=2048, expansion_size=expansion_size) # Thêm Expansion Block
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, attention_size, dropout_rate)

    def forward(self, images, captions, caption_lengths):
        """
        Forward pass của mô hình.

        Args:
            images (torch.Tensor): Tensor chứa ảnh đầu vào.
            captions (torch.Tensor): Tensor chứa các chú thích đã mã hóa.
            caption_lengths (list): List chứa độ dài của các chú thích.

        Returns:
            tuple: (logits, attention weights).
        """
        features = self.encoder(images)
        features = self.expansion_block(features) # Áp dụng Expansion Block
        logits, att_weights = self.decoder(features, captions, caption_lengths)
        return logits, att_weights

# --------------------------------------------------------------------------------
# 4. Huấn luyện mô hình
# --------------------------------------------------------------------------------

def train(model, dataloader, criterion, optimizer, clip_grad, epoch):
    """
    Huấn luyện mô hình trong một epoch.

    Args:
        model (nn.Module): Mô hình cần huấn luyện.
        dataloader (DataLoader): DataLoader cho dữ liệu huấn luyện.
        criterion (nn.Module): Hàm loss.
        optimizer (optim.Optimizer): Optimizer.
        clip_grad (float): Giá trị để clip gradient.
        epoch (int): Số epoch hiện tại.
    """
    model.train()
    total_loss = 0
    for i, (images, captions, caption_lengths) in enumerate(dataloader):
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)
        targets = captions[:, 1:]  # Loại bỏ token <start>
        outputs, _ = model(images, captions, caption_lengths)
        loss = criterion(outputs.view(-1, outputs.size(2)), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        # Clip gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        total_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch}/{NUM_EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch}/{NUM_EPOCHS}], Average Loss: {avg_loss:.4f}')
    return avg_loss

# --------------------------------------------------------------------------------
# 5. Đánh giá mô hình
# --------------------------------------------------------------------------------

def generate_caption(model, image, vocab, max_length=50):
    """
    Tạo chú thích cho một ảnh.

    Args:
        model (nn.Module): Mô hình đã được huấn luyện.
        image (torch.Tensor): Ảnh đầu vào.
        vocab (dict): Dictionary ánh xạ từ tới chỉ số.
        max_length (int): Độ dài tối đa của chú thích được tạo ra.

    Returns:
        list: List các từ trong chú thích đã được tạo ra.
    """
    model.eval()
    with torch.no_grad():
        features = model.encoder(image.unsqueeze(0).to(DEVICE))
        features = model.expansion_block(features) # Áp dụng Expansion Block
        hidden = None
        # Khởi tạo input với token <start>
        input_token = torch.tensor([vocab['<start>']], dtype=torch.long).to(DEVICE)
        caption = []
        for _ in range(max_length):
            if hidden is None:
                context, _ = model.decoder.attention(features, torch.zeros(1, model.decoder.hidden_size).to(DEVICE))
            else:
                context, _ = model.decoder.attention(features, hidden[0])
            input_combined = torch.cat([context, model.decoder.embed(input_token)], dim=1)
            hidden = model.decoder.lstm(input_combined, hidden)
            output = model.decoder.dropout(hidden[0])
            logit = model.decoder.fc(output)
            _, predicted_index = torch.max(logit, dim=1)
            predicted_word = [word for word, index in vocab.items() if index == predicted_index.item()][0]
            caption.append(predicted_word)
            input_token = predicted_index
            if predicted_word == '<end>':
                break
        return caption

def evaluate(model, dataloader, vocab):
    """
    Đánh giá mô hình trên tập test bằng các độ đo BLEU, CIDEr.

    Args:
        model (nn.Module): Mô hình đã được huấn luyện.
        dataloader (DataLoader): DataLoader cho dữ liệu test.
        vocab (dict): Dictionary ánh xạ từ tới chỉ số.

    Returns:
        dict: Dictionary chứa các độ đo đánh giá (BLEU-1, BLEU-2, BLEU-3, BLEU-4, CIDEr).
    """
    model.eval()
    references = []
    predictions = []
    for images, _, image_ids in dataloader: # Lấy image_ids từ dataloader
        images = images.to(DEVICE)
        for i in range(images.size(0)): # Lặp qua từng ảnh trong batch
            image_id = dataloader.dataset.image_ids[dataloader.dataset.image_ids.index(image_ids[i])] # Lấy ID của ảnh
            caption = generate_caption(model, images[i], vocab)
            predicted_caption = ' '.join(caption)
            predictions.append(predicted_caption.split())

            # Lấy các chú thích gốc cho ảnh
            ground_truth_captions = dataloader.dataset.captions[image_id]
            ground_truth_captions = [nltk.word_tokenize(caption) for caption in ground_truth_captions]
            references.append(ground_truth_captions)

    # Tính BLEU scores
    smoothing = SmoothingFunction().method1
    bleu1 = corpus_bleu(references, predictions, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = corpus_bleu(references, predictions, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = corpus_bleu(references, predictions, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(references, predictions, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

    # Tính CIDEr score
    scorer = Cider()
    (cider, _) = scorer.compute_score(references, predictions)

    print(f'BLEU-1: {bleu1:.4f}')
    print(f'BLEU-2: {bleu2:.4f}')
    print(f'BLEU-3: {bleu3:.4f}')
    print(f'BLEU-4: {bleu4:.4f}')
    print(f'CIDEr: {cider:.4f}')
    return {
        'BLEU-1': bleu1,
        'BLEU-2': bleu2,
        'BLEU-3': bleu3,
        'BLEU-4': bleu4,
        'CIDEr': cider
    }

# --------------------------------------------------------------------------------
# 6. Main
# --------------------------------------------------------------------------------

def main():
    """
    Hàm main để chạy toàn bộ quá trình:
    1. Tiền xử lý dữ liệu
    2. Xây dựng từ vựng
    3. Chia dữ liệu train/test
    4. Tạo DataLoader
    5. Khởi tạo mô hình, hàm loss, optimizer
    6. Huấn luyện mô hình
    7. Đánh giá mô hình
    """
    # 1. Tiền xử lý dữ liệu
    captions = load_captions(CAPTION_FILE)
    captions = preprocess_captions(captions)

    # 2. Xây dựng từ vựng
    vocab = build_vocabulary(captions)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # 3. Chia dữ liệu train/test
    train_data, test_data = split_data(captions, TRAIN_RATIO)

    # 4. Tạo DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = Flickr8kDataset(IMAGE_DIR, train_data, vocab, transform)
    test_dataset = Flickr8kDataset(IMAGE_DIR, test_data, vocab, transform) # Pass test_data
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,collate_fn=collate_fn) # Pass test_dataset

    # 5. Khởi tạo mô hình, hàm loss, optimizer
    model = ImageCaptioningModel(EMBED_SIZE, HIDDEN_SIZE, vocab_size, ATTENTION_SIZE, DROPOUT_RATE, EXPANSION_SIZE).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Huấn luyện mô hình
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_loader, criterion, optimizer, CLIP_GRAD, epoch)

        # Lưu checkpoint sau mỗi epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'vocab': vocab
        }, f'checkpoint_epoch_{epoch}.pth')
    print("Finished Training")

    # 7. Đánh giá mô hình
    print("Evaluating on Test Set:")
    evaluation_metrics = evaluate(model, test_loader, vocab)
    print(evaluation_metrics)

if __name__ == "__main__":
    main()
