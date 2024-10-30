import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformer import EnhancedTimeSeriesTransformer
from dataloader import create_dataloader
import matplotlib.pyplot as plt
import argparse

def compute_soft_assignment(features, cluster_centers, alpha=1.0):
    dist = torch.cdist(features, cluster_centers, p=2)
    q = (1 + dist**2 / alpha).pow(-(alpha + 1) / 2)
    q = q / (q.sum(dim=1, keepdim=True) + 1e-8)
    return q

def compute_target_distribution(q):
    p = q**2 / (q.sum(dim=0, keepdim=True) + 1e-8)
    p = p / (p.sum(dim=1, keepdim=True) + 1e-8)
    return p

def plot_loss_curve(losses):
    """Plot the training loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Loss', linewidth=0.8)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.legend()
    plt.show()

def debug_data_loader(data_loader):
    """Randomly print a batch of data to check correctness."""
    print("\n[DEBUG] Checking DataLoader...")
    for batch_idx, data in enumerate(data_loader):
        if batch_idx % 10 == 0:
            print(f"\nBatch {batch_idx}:")
            print(data.shape)
            print(data[0])  # Print the first sample in the batch
        if batch_idx > 6:
            break
    print("[DEBUG] DataLoader check completed.\n")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedTimeSeriesTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Load training data from specified path
    train_dataloader = create_dataloader(args.data_path, batch_size=32, is_test=False)

    # Debugging: Check data loader by printing sample data
    if args.debug:
        debug_data_loader(train_dataloader)

    n_clusters = args.n_clusters
    cluster_centers = torch.rand(n_clusters, 128, device=device)
    cluster_centers.requires_grad = True

    num_epochs = 20
    beta = 0.1

    data_projection = nn.Linear(2, 128).to(device)

    # Store batch losses for plotting
    batch_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, data in enumerate(train_dataloader):
            data = data.to(device)

            optimizer.zero_grad()

            features = model(data)
            projected_data = data_projection(data)
            projected_data = projected_data.mean(dim=1)

            recon_loss = F.mse_loss(features, projected_data)

            q = compute_soft_assignment(features, cluster_centers)
            p = compute_target_distribution(q)

            cluster_loss = F.kl_div(q.log(), p, reduction='batchmean')

            loss = recon_loss + beta * cluster_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            with torch.no_grad():
                cluster_centers -= cluster_centers.grad * 0.01

            epoch_loss += loss.item()
            batch_losses.append(loss.item())  # Record batch loss

            if args.debug and batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item()}")

        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {epoch_loss/len(train_dataloader)}")

    # Plot the loss curve after training
    plot_loss_curve(batch_losses)

    # Save the trained model and cluster centers
    torch.save(model.state_dict(), 'transformer_model.pth')
    torch.save(cluster_centers, 'cluster_centers.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Time Series Transformer with Clustering')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters for K-means')
    parser.add_argument('--data_path', type=str, default='data/train', help='Path to training data')
    parser.add_argument('--debug', action='store_true', help='Enable debugging mode to print sample data')
    args = parser.parse_args()

    main(args)

