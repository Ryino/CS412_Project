import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_raw_data(file_path):
    """Load the raw IEMOCAP data."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print("Data type:", type(data))
    
    if isinstance(data, list):
        print("Data is a list with length:", len(data))
        print("\nFirst item in data:")
        if len(data) > 0:
            first_dict = data[0]
            if isinstance(first_dict, dict):
                first_session_id = next(iter(first_dict.keys()))
                first_session_data = first_dict[first_session_id]
                print(f"\nSession ID: {first_session_id}")
                print(f"Session data type: {type(first_session_data)}")
                if isinstance(first_session_data, list):
                    print(f"Session data length: {len(first_session_data)}")
                    print("\nFirst few items in session data:")
                    for i, item in enumerate(first_session_data[:10]):
                        print(f"\nItem {i}:")
                        print(f"Type: {type(item)}")
                        if isinstance(item, (list, np.ndarray)):
                            print(f"Length: {len(item)}")
                            if len(item) > 0:
                                print(f"First element type: {type(item[0])}")
                                if isinstance(item[0], str):
                                    print(f"First element: {item[0]}")
                                else:
                                    print(f"First element length: {len(item[0]) if isinstance(item[0], (list, np.ndarray)) else 'scalar'}")
                        else:
                            print(f"Value: {item}")
                            
                    # Print the last few items to see if they contain features
                    print("\nLast few items in session data:")
                    for i, item in enumerate(first_session_data[-5:]):
                        print(f"\nItem {len(first_session_data)-5+i}:")
                        print(f"Type: {type(item)}")
                        if isinstance(item, (list, np.ndarray)):
                            print(f"Length: {len(item)}")
                            if len(item) > 0:
                                print(f"First element type: {type(item[0])}")
                                if isinstance(item[0], str):
                                    print(f"First element: {item[0]}")
                                else:
                                    print(f"First element length: {len(item[0]) if isinstance(item[0], (list, np.ndarray)) else 'scalar'}")
                        else:
                            print(f"Value: {item}")
    
    return data

def extract_features(data):
    """Extract and preprocess features from the raw data."""
    processed_data = []
    
    if isinstance(data, list):
        # Process each dictionary in the list
        for data_dict in data:
            if isinstance(data_dict, dict):
                # Process each session in the dictionary
                for session_id, session_data in data_dict.items():
                    try:
                        # Skip if session_data is not a list
                        if not isinstance(session_data, list):
                            continue
                            
                        # Get utterance IDs (first half of the list)
                        mid_point = len(session_data) // 2
                        utterance_ids = session_data[:mid_point]
                        feature_data = session_data[mid_point:]
                        
                        # Process each feature item
                        for i, features in enumerate(feature_data):
                            try:
                                # Skip if not a valid feature array
                                if not isinstance(features, (list, np.ndarray)):
                                    continue
                                    
                                # Convert features to numpy array
                                features_array = np.array(features, dtype=np.float32)
                                
                                # Skip if we don't have enough features
                                if len(features_array) < 410:  # 300 + 74 + 35 + 1 (label)
                                    continue
                                
                                # Extract features
                                text_features = features_array[:300]
                                audio_features = features_array[300:374]
                                vision_features = features_array[374:409]
                                label = int(features_array[409])
                                
                                # Validate features
                                if np.isnan(text_features).any() or np.isnan(audio_features).any() or np.isnan(vision_features).any():
                                    continue
                                    
                                # Get corresponding utterance ID
                                utterance_id = utterance_ids[i] if i < len(utterance_ids) else 'unknown'
                                
                                processed_data.append({
                                    'session_id': session_id,
                                    'utterance_id': utterance_id,
                                    'text': text_features,
                                    'audio': audio_features,
                                    'vision': vision_features,
                                    'label': label
                                })
                            except (IndexError, ValueError, TypeError) as e:
                                continue
                    except Exception as e:
                        print(f"Error processing session {session_id}: {str(e)}")
                        continue
    else:
        raise ValueError(f"Expected list of dictionaries, got {type(data)}")
    
    if not processed_data:
        raise ValueError("No valid data was processed")
        
    print(f"Successfully processed {len(processed_data)} samples")
    return processed_data

def normalize_features(data):
    """Normalize features using StandardScaler."""
    if not data:
        raise ValueError("No data to normalize")
        
    # Extract features
    text_features = np.array([item['text'] for item in data])
    audio_features = np.array([item['audio'] for item in data])
    vision_features = np.array([item['vision'] for item in data])
    
    print(f"Feature shapes before normalization:")
    print(f"Text: {text_features.shape}, Audio: {audio_features.shape}, Vision: {vision_features.shape}")
    
    # Check for NaN or infinite values
    for name, features in [('text', text_features), ('audio', audio_features), ('vision', vision_features)]:
        if np.isnan(features).any():
            print(f"Warning: NaN values found in {name} features")
        if np.isinf(features).any():
            print(f"Warning: Infinite values found in {name} features")
    
    # Initialize scalers
    text_scaler = StandardScaler()
    audio_scaler = StandardScaler()
    vision_scaler = StandardScaler()
    
    # Fit and transform features
    text_features = text_scaler.fit_transform(text_features)
    audio_features = audio_scaler.fit_transform(audio_features)
    vision_features = vision_scaler.fit_transform(vision_features)
    
    # Update data with normalized features
    for i, item in enumerate(data):
        item['text'] = text_features[i]
        item['audio'] = audio_features[i]
        item['vision'] = vision_features[i]
    
    return data

def split_data(data, test_size=0.2, val_size=0.1, random_state=42):
    """Split data into train, validation, and test sets."""
    # First split into train+val and test
    train_val, test = train_test_split(data, test_size=test_size, random_state=random_state)
    
    # Then split train+val into train and val
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=random_state)
    
    return train, val, test

def save_processed_data(train, val, test, output_dir='processed_data'):
    """Save processed data to pickle files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert to dictionary format expected by the model
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
    
    # Save metadata for reference
    metadata = {
        'train_sessions': [(item['session_id'], item['utterance_id']) for item in train],
        'valid_sessions': [(item['session_id'], item['utterance_id']) for item in val],
        'test_sessions': [(item['session_id'], item['utterance_id']) for item in test]
    }
    
    # Save to pickle files
    output_path = os.path.join(output_dir, 'IEMOCAP_processed.pkl')
    metadata_path = os.path.join(output_dir, 'IEMOCAP_metadata.pkl')
    
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Processed data saved to {output_path}")
    print(f"Metadata saved to {metadata_path}")

def main():
    # Load raw data
    raw_data = load_raw_data('IEMOCAP_features.pkl')
    
    # Extract features
    processed_data = extract_features(raw_data)
    
    # Normalize features
    processed_data = normalize_features(processed_data)
    
    # Split data
    train, val, test = split_data(processed_data)
    
    # Save processed data
    save_processed_data(train, val, test)
    
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