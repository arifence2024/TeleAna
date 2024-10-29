import torch
import torch.optim as optim
import torch.nn.functional as F
from transformer import EnhancedTimeSeriesTransformer
from dataloader import create_dataloader
import argparse

def compute_soft_assignment(features, cluster_centers, alpha=1.0):
    dist = torch.cdist(features, cluster_centers, p=2)
    q = (1 + dist**2 / alpha).pow(-(alpha + 1) / 2)
    q = q / q.sum(dim=1, keepdim=True)
    return q

def compute_target_distribution(q):
    p = q**2 / q.sum(dim=0, keepdim=True)
    p = p / p.sum(dim=1, keepdim=True)
    return p

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedTimeSeriesTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load training data
    train_dataloader = create_dataloader('/home/hu/Downloads/LOGDATA', batch_size=32, is_test=False)

    n_clusters = args.n_clusters
    cluster_centers = torch.rand(n_clusters, 128, device=device)
    cluster_centers.requires_grad = True

    num_epochs = 10
    beta = 0.1

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for data in train_dataloader:
            data = data.to(device).float()
            optimizer.zero_grad()

            # Extract features
            features = model(data)
            # Compute soft assignments
            q = compute_soft_assignment(features, cluster_centers)
            # Compute target distribution
            p = compute_target_distribution(q)

            # Calculate reconstruction loss
            recon_loss = F.mse_loss(features, data.mean(dim=1))
            # Calculate clustering loss
            cluster_loss = F.kl_div(q.log(), p, reduction='batchmean')

            # Total loss
            loss = recon_loss + beta * cluster_loss
            loss.backward()

            optimizer.step()
            with torch.no_grad():
                cluster_centers -= cluster_centers.grad * 0.01

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_dataloader)}")

    torch.save(model.state_dict(), 'transformer_model.pth')
    torch.save(cluster_centers, 'cluster_centers.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Time Series Transformer with Clustering')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters for K-means')
    args = parser.parse_args()

    main(args)
