import torch
import torch.nn as nn
import timm
from tqdm import tqdm
from prepare_data import train_loader, val_loader, NUM_CLASSES, CLASS_NAMES

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 30
LR = 1e-4

# Load pretrained EfficientNet-B0, replace classifier head
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# Two-phase training: freeze backbone first, then unfreeze
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.set_grad_enabled(train):
        for imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            loss = criterion(out, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (out.argmax(1) == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total

best_val_acc = 0
for epoch in range(1, EPOCHS + 1):
    # Unfreeze full model after 5 warmup epochs
    if epoch == 6:
        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=LR / 10)
        print('Backbone unfrozen — fine-tuning all layers')

    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss,   val_acc   = run_epoch(val_loader,   train=False)
    scheduler.step()

    print(f'Epoch {epoch:02d} | '
          f'Train loss: {train_loss:.4f} acc: {train_acc:.3f} | '
          f'Val loss: {val_loss:.4f} acc: {val_acc:.3f}')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'  Saved best model (val acc: {val_acc:.3f})')

print(f'\nBest validation accuracy: {best_val_acc:.3f}')