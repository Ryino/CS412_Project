import pickle
import numpy as np
import torch
import torch.nn.functional as F

# Load the dataset
with open('IEMOCAP_features.pkl', 'rb') as f:
    data = pickle.load(f)

print("Dataset type:", type(data))
if isinstance(data, list):
    print("Number of items in list:", len(data))
    
    # Check the first item in detail
    print("\nFirst item type:", type(data[0]))
    if isinstance(data[0], (list, tuple)):
        print("First item length:", len(data[0]))
        print("First item content:", data[0])
        
        # Try to convert first item to numpy array
        try:
            arr = np.array(data[0], dtype=np.float32)
            print("\nConverted to numpy array:")
            print("Shape:", arr.shape)
            print("Dtype:", arr.dtype)
            print("First few values:", arr[:5])
        except Exception as e:
            print("Could not convert to numpy array:", e)
            
    # Check if we have labels
    if len(data[0]) > 0:
        print("\nLast element of first item:", data[0][-1])
        print("Type of last element:", type(data[0][-1]))
        
    # Print some statistics about the data
    print("\nData statistics:")
    print("Number of unique values in first position:", len(set(item[0] for item in data)))
    print("Number of unique values in last position:", len(set(item[-1] for item in data)))
    
    # Try to find label distribution
    try:
        labels = [item[-1] for item in data]
        unique_labels = set(labels)
        print("\nUnique labels:", unique_labels)
        print("Label distribution:")
        for label in unique_labels:
            count = sum(1 for x in labels if x == label)
            print(f"Label {label}: {count} samples")
    except Exception as e:
        print("Could not analyze labels:", e)

# Check train data
train_data = data.get('train', data.get('Train', None))
if isinstance(train_data, dict):
    print("\nTrain data format: Dictionary")
    print("Available keys:", list(train_data.keys()))
    print("Text shape:", train_data['text'].shape)
    print("Audio shape:", train_data['audio'].shape)
    print("Vision shape:", train_data['vision'].shape)
    print("Labels shape:", train_data['labels'].shape)
    print("Unique labels:", np.unique(train_data['labels']))
elif isinstance(train_data, list):
    print("\nTrain data format: List")
    print("Number of samples:", len(train_data))
    if len(train_data) > 0:
        print("\nFirst sample structure:")
        for i, item in enumerate(train_data[0]):
            print(f"Item {i} type:", type(item))
            if isinstance(item, (list, tuple)):
                print(f"Item {i} length:", len(item))
                if len(item) > 0:
                    print(f"Item {i} first element type:", type(item[0]))
        
        # Check for labels in first few samples
        print("\nChecking for labels in first 5 samples:")
        for i in range(min(5, len(train_data))):
            print(f"Sample {i} length:", len(train_data[i]))
            if len(train_data[i]) > 3:
                print(f"Sample {i} label:", train_data[i][3])
else:
    print("\nTrain data format:", type(train_data))

# After batch = next(data_iter)
for tensor, name in zip([text, acoustic, visual, label], ['text', 'acoustic', 'visual', 'label']):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN or Inf detected in {name}")
        print(tensor)
        raise ValueError(f"NaN or Inf detected in {name}")

F.normalize(tensor, p=2, dim=-1, eps=1e-8)

shared_embs, diff_embs = model.forward(text, acoustic, visual)
for tensor, name in zip([shared_embs, diff_embs], ['shared_embs', 'diff_embs']):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN or Inf detected in {name}")
        print(tensor)
        raise ValueError(f"NaN or Inf detected in {name}")

if torch.isnan(pred).any() or torch.isinf(pred).any():
    print("NaN or Inf in predictions!")
    print(pred)
    raise ValueError("NaN or Inf in predictions")

if torch.isnan(loss) or torch.isinf(loss):
    print("NaN or Inf detected in loss")
    print("Pred:", pred)
    print("Label:", label)
    raise ValueError("NaN or Inf detected in loss")

total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5
print("Gradient norm:", total_norm)

print("Unique labels:", label.unique())
print("Label dtype:", label.dtype)
print("Pred shape:", pred.shape)
print("Label shape:", label.shape) 