from jiwer import wer, cer
import torch.nn as nn
import torch
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import DataLoader
import torchaudio.transforms as T
from tqdm import tqdm  # Import tqdm for progress monitoring
import subprocess  # Import subprocess for running system commands
import time  # Import time for sleep functionality
import threading

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define data path and dataset
train_data_path = "C:\\Users\\MOHAMED HAMAD\\Desktop\\gggg\\"
train_dataset = LIBRISPEECH(
    train_data_path, url='train-clean-100', download=False)

# Use only 1/20 of the dataset
num_samples = len(train_dataset)
reduced_dataset = torch.utils.data.Subset(
    train_dataset, range(num_samples // 20))

# Define a data loader for batching


def collate_fn(batch):
    audio_signals, labels, input_lengths, label_lengths = [], [], [], []

    for waveform, sample_rate, utterance, _, _, _ in batch:
        # Resample to 16kHz
        waveform = T.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        # Add to the lists
        audio_signals.append(waveform.squeeze(0))
        # Simple character-level encoding
        labels.append(torch.tensor(
            [ord(c) - 96 for c in utterance.lower() if c.isalpha()]))
        input_lengths.append(waveform.size(1))
        label_lengths.append(len(labels[-1]))

    # Padding
    audio_signals = torch.nn.utils.rnn.pad_sequence(
        audio_signals, batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return audio_signals, labels, torch.tensor(input_lengths), torch.tensor(label_lengths)


# Reduced batch size to manage memory
train_loader = DataLoader(reduced_dataset, batch_size=1,
                          shuffle=True, collate_fn=collate_fn)


class ASRModel(nn.Module):
    def _init_(self, input_dim=16000, hidden_dim=256, num_classes=29):
        super(ASRModel, self)._init_()

        # CNN Layer
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # RNN (GRU)
        self.gru = nn.GRU(64, hidden_dim, num_layers=2,
                          bidirectional=True, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        # Input: [batch, time, channels] -> [batch, channels, time]
        x = x.unsqueeze(1)  # [batch, 1, time]

        # Apply CNN
        x = self.conv(x)

        # Reduce the lengths after each convolution (assuming stride of 2)
        lengths = lengths // 2  # After first conv layer
        lengths = lengths // 2  # After second conv layer

        # Prepare for RNN input (swap axes for RNN [batch, time, features])
        x = x.transpose(1, 2).contiguous()  # Ensure contiguous tensor

        # Pack sequence for RNN
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(),  # Move lengths to CPU
            batch_first=True,
            enforce_sorted=False
        )

        # Forward through GRU
        packed_output, _ = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )

        # Pass through fully connected layer
        output = self.fc(output)

        return output, lengths


# Define CTC loss with blank label
ctc_loss = nn.CTCLoss(blank=28)

# Initialize the model, optimizer, and loss function
model = ASRModel().to(device)  # Move model to GPU
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Function to run nvidia-smi command


def monitor_gpu():
    while True:
        subprocess.run(["nvidia-smi"])
        time.sleep(250)  # Sleep for 5 seconds before the next check

# Function to train the model with progress monitoring


def train(model, loader, optimizer, loss_fn, epochs=10):
    model.train()

    # Start monitoring GPU in a separate thread
    gpu_monitor_thread = threading.Thread(target=monitor_gpu)
    # Allows the thread to exit when main program does
    gpu_monitor_thread.daemon = True
    gpu_monitor_thread.start()

    for epoch in range(epochs):
        running_loss = 0.0
        # Create a progress bar
        progress_bar = tqdm(
            loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch')

        for batch_idx, (inputs, targets, input_lengths, target_lengths) in enumerate(progress_bar):
            optimizer.zero_grad()

            # Move data to GPU
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            # Forward pass
            log_probs, input_lengths = model(
                inputs.contiguous(), input_lengths)  # Ensure contiguous

            # Compute CTC Loss
            loss = loss_fn(log_probs.permute(1, 0, 2), targets,
                           input_lengths, target_lengths)

            # Backprop and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Update the progress bar with the current loss
            progress_bar.set_postfix(loss=loss.item())

        print(
            f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(loader):.4f}')


# Train the model
train(model, train_loader, optimizer, ctc_loss, epochs=10)

# Evaluation function


def evaluate(model, loader):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets, input_lengths, target_lengths in loader:
            inputs = inputs.to(device)  # Move inputs to GPU
            log_probs, _ = model(inputs.contiguous(),
                                 input_lengths)  # Ensure contiguous
            log_probs = torch.nn.functional.log_softmax(log_probs, dim=-1)

            # Get predicted indices (using greedy decoding here)
            pred_indices = torch.argmax(log_probs, dim=-1)

            # Convert predictions and targets to a flat list for CER and WER calculation
            all_preds.extend(pred_indices.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Compute CER and WER
    cer_score = cer(all_targets, all_preds)
    wer_score = wer(all_targets, all_preds)

    print(f'CER: {cer_score:.4f}, WER: {wer_score:.4f}')


# Evaluate the model (on validation set or a sample)
# Replace train_loader with validation loader if available
evaluate(model, train_loader)