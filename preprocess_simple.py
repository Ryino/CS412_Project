import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def main():
    # Load raw data
    print("Loading raw data...")
    data_path = r"D:\Documents\4th Year\CS 412\MMCL\mosei.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Type of data: {type(data)}")
    if isinstance(data, list):
        print(f"Length of data: {len(data)}")
        if len(data) > 0:
            print(f"Type of first element: {type(data[0])}")
            first_dict = data[0]
            print(f"First element keys: {list(first_dict.keys())}")
            for k, v in first_dict.items():
                print(f"Key: {k}")
                print(f"Type of value: {type(v)}")
                print(f"Length of value: {len(v)}")
                print(f"First 3 items in value: {v[:3]}")
                if len(v) > 0:
                    print(f"Type of first value item: {type(v[0])}")
                    print(f"First value item: {v[0]}")
                break
        if len(data) > 1:
            print(f"\nType of second element: {type(data[1])}")
            print(f"Second element: {data[1]}")
    elif isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        first_key = list(data.keys())[0]
        print(f"Type of first value: {type(data[first_key])}")
        print(f"First value: {data[first_key]}")
    else:
        print("Unknown data structure!")
    
    # Process data
    print("Processing data...")
    processed_data = []
    
    # Handle the list structure
    for session_data in data:
        if isinstance(session_data, list) and len(session_data) >= 2:
            utterance_ids = session_data[0]  # First element contains utterance IDs
            feature_data = session_data[1]   # Second element contains features
            
            for i, features in enumerate(feature_data):
                if isinstance(features, (list, np.ndarray)):
                    features_array = np.array(features, dtype=np.float32)
                    if len(features_array) >= 410:
                        text_features = features_array[:300]
                        audio_features = features_array[300:374]
                        vision_features = features_array[374:409]
                        label = int(features_array[409])
                        
                        if not (np.isnan(text_features).any() or np.isnan(audio_features).any() or np.isnan(vision_features).any()):
                            utterance_id = utterance_ids[i] if i < len(utterance_ids) else 'unknown'
                            processed_data.append({
                                'utterance_id': utterance_id,
                                'text': text_features,
                                'audio': audio_features,
                                'vision': vision_features,
                                'label': label
                            })
    
    print(f"Successfully processed {len(processed_data)} samples")
    
    # Normalize features
    print("Normalizing features...")
    text_features = np.array([item['text'] for item in processed_data])
    audio_features = np.array([item['audio'] for item in processed_data])
    vision_features = np.array([item['vision'] for item in processed_data])
    
    text_scaler = StandardScaler()
    audio_scaler = StandardScaler()
    vision_scaler = StandardScaler()
    
    text_features = text_scaler.fit_transform(text_features)
    audio_features = audio_scaler.fit_transform(audio_features)
    vision_features = vision_scaler.fit_transform(vision_features)
    
    for i, item in enumerate(processed_data):
        item['text'] = text_features[i]
        item['audio'] = audio_features[i]
        item['vision'] = vision_features[i]
    
    # Split data
    print("Splitting data...")
    train_val, test = train_test_split(processed_data, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1/(1-0.2), random_state=42)
    
    # Save data
    print("Saving data...")
    output_dir = 'processed_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    processed_data = {
        'train': {
            'text': np.array([item['text'] for item in train]),
            'audio': np.array([item['audio'] for item in train]),
            'vision': np.array([item['vision'] for item in train]),
            'labels': np.array([item['label'] for item in train])
        },
        'valid': {
            'text': np.array([item['text'] for item in val]),
            'audio': np.array([item['audio'] for item in val]),
            'vision': np.array([item['vision'] for item in val]),
            'labels': np.array([item['label'] for item in val])
        },
        'test': {
            'text': np.array([item['text'] for item in test]),
            'audio': np.array([item['audio'] for item in test]),
            'vision': np.array([item['vision'] for item in test]),
            'labels': np.array([item['label'] for item in test])
        }
    }
    
    metadata = {
        'train_sessions': [item['utterance_id'] for item in train],
        'valid_sessions': [item['utterance_id'] for item in val],
        'test_sessions': [item['utterance_id'] for item in test]
    }
    
    output_path = os.path.join(output_dir, 'IEMOCAP_processed.pkl')
    metadata_path = os.path.join(output_dir, 'IEMOCAP_metadata.pkl')
    
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Processed data saved to {output_path}")
    print(f"Metadata saved to {metadata_path}")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(processed_data)}")
    print(f"Train samples: {len(train)}")
    print(f"Validation samples: {len(val)}")
    print(f"Test samples: {len(test)}")
    
    # Print label distribution
    for split_name, split_data in [('Train', train), ('Validation', val), ('Test', test)]:
        labels = [item['label'] for item in split_data]
        unique_labels = np.unique(labels)
        print(f"\n{split_name} set label distribution:")
        for label in unique_labels:
            count = sum(1 for x in labels if x == label)
            print(f"Label {label}: {count} samples")

if __name__ == '__main__':
    main() 