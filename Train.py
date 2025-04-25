from Model import Transformer
from Config import config
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torch.nn.parallel import DataParallel

# Main
def train(model_path):
    device = get_device()
    print(f"Using device: {device}")

    # Load components
    de_tokenizer = AutoTokenizer.from_pretrained(config.de_tokenizer_name)
    en_tokenizer = AutoTokenizer.from_pretrained(config.en_tokenizer_name)
    training_data = get_dataset()
    training_generator = get_dataloader(training_data)
    
    # Initialize model with multi-GPU support
    transformer = get_model(de_tokenizer, en_tokenizer, device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = get_optimizer(transformer)

    # Training loop
    transformer.train()
    for epoch in range(config.epochs):
        start_time = time.time()
        epoch_loss = 0
        optimizer.zero_grad()

        for step, data in enumerate(training_generator):
            batch_de_tokens = get_batch_tokens(data['translation']['de'], de_tokenizer, device)
            batch_en_tokens = get_batch_tokens(data['translation']['en'], en_tokenizer, device)

            # Forward pass
            output = transformer(batch_de_tokens, batch_en_tokens[:, :-1])
            
            # Loss calculation
            loss = criterion(
                output.contiguous().view(-1, en_tokenizer.vocab_size),
                batch_en_tokens[:, 1:].contiguous().view(-1)
            )
            
            # Backward pass with gradient accumulation
            loss.backward()

            # Gradient accumulation steps
            if (step + 1) % config.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Training stats
            epoch_loss += loss.item()
            if step % 100 == 0:
                print(f'Step {step}, Loss: {loss.item():.4f}, Time: {time.time() - start_time:.2f}s')

            # Save checkpoint
            if step % 1000 == 0:
                save_model(model_path, epoch, transformer, optimizer)

        # Epoch statistics
        avg_epoch_loss = epoch_loss / len(training_generator)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}, Avg Loss: {avg_epoch_loss:.4f}, Time: {epoch_time:.2f}s")

        # Save model after each epoch
        save_model(model_path, epoch, transformer, optimizer)

    # Final save
    save_model(model_path, epoch, transformer, optimizer)

def get_device():
    if torch.cuda.is_available():
        # Use all available GPUs
        return torch.device('cuda' if torch.cuda.device_count() == 1 else 'cuda:0')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_dataset():
    return load_dataset('wmt/wmt14', 'de-en', split='train')

def get_dataloader(dataset):
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )

def get_model(de_tokenizer, en_tokenizer, device):
    model = Transformer(
        src_vocab_size=de_tokenizer.vocab_size,
        tgt_vocab_size=en_tokenizer.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout,
        device=device
    )
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = DataParallel(model)
    
    return model.to(device)

def get_optimizer(transformer):
    return optim.Adam(
        transformer.parameters(),
        lr=config.lr,
        betas=(0.9, 0.98),
        eps=1e-9
    )

def get_batch_tokens(data, tokenizer, device):
    return torch.stack([
        torch.tensor(ids, device=device) 
        for ids in tokenizer(
            data,
            truncation=True,
            padding='max_length',
            max_length=config.max_seq_length,
            return_tensors='np'
        ).input_ids
    ])

def save_model(model_path, epoch, transformer, optimizer):
    full_model_path = f"{model_path}_epoch{epoch+1}.pt"
    # Handle DataParallel wrapper
    model_state = transformer.module.state_dict() if isinstance(transformer, DataParallel) else transformer.state_dict()
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict()
    }, full_model_path)

if __name__ == "__main__":
    train(config.model_path)
