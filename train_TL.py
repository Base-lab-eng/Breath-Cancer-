from utils_py import *
from data_setup import setup
from Transfer_Learning import build_model
from engine import train_one_epoch, validate
from Transfer_Learning_2 import build_model_2
from Transfer_Learning_3 import build_model_3

model_type='mobilenet_v2'  # 'resnet50' or 'mobilenet_v2'

if model_type == 'resnet50':
    model = build_model()
    
elif model_type == 'mobilenet_v2':
    model = build_model_2()
    
elif model_type=='EfficientNet_B0':
    model = build_model_3()
    



train_loader, val_loader, test_loader, train_labels = setup()

class_counts = np.bincount(train_labels)
weights = class_counts.max() / class_counts
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5
)

EPOCHS = 30
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc     = validate(model, val_loader, criterion, device)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, optimizer, epoch, f"checkpoints/best_model_{model_type}.pth")

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")