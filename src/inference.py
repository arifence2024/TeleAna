import torch
import sys
from transformer import EnhancedTimeSeriesTransformer
from dataloader import PerformanceDataset

def load_model(model_path, device):
    """
    Loads the trained model from a file.

    :param model_path: Path to the model file
    :param device: Device to load the model on
    :return: Loaded model
    """
    model = EnhancedTimeSeriesTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def inference(input_file, model, device):
    """
    Performs inference on a single input file.

    :param input_file: Path to the input CSV file
    :param model: Trained Transformer model
    :param device: Device to perform inference on
    """
    dataset = PerformanceDataset(root_dir=input_file, is_test=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device).float()
            features = model(data)
            print("Inference output:", features.cpu().numpy())

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python inference.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('transformer_model.pth', device)
    inference(input_file, model, device)

