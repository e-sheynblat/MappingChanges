import torch.optim as optim
from torch.utils.data import DataLoader

def train_siamese(model, dataset, epochs=5, batch_size=4, lr=1e-4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()  # binary change/no-change
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for hist, curr, label in loader:
            optimizer.zero_grad()
            output = model(hist, curr)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {running_loss/len(loader):.4f}")
