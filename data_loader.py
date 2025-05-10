import numpy as np
import pickle
import torch
import torch.utils.data as data


class Data(data.Dataset):
    def __init__(self, path, mode='train'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle different dataset split naming conventions
        if mode == 'train':
            dataset = data.get('train', data.get('Train', None))
        elif mode == 'valid':
            dataset = data.get('valid', data.get('Valid', data.get('dev', data.get('Dev', None))))
        else:  # test
            dataset = data.get('test', data.get('Test', None))
            
        if dataset is None:
            raise ValueError(f"Could not find {mode} split in dataset. Available keys: {list(data.keys())}")

        # Handle both dict and list-of-tuples formats
        if isinstance(dataset, dict):
            # Convert to float32 and handle inf values
            text = dataset['text'].astype(np.float32)
            text[text == -np.inf] = 0
            text[text == np.inf] = 0
            text = np.nan_to_num(text, 0)
            
            audio = dataset['audio'].astype(np.float32)
            audio[audio == -np.inf] = 0
            audio[audio == np.inf] = 0
            audio = np.nan_to_num(audio, 0)
            
            vision = dataset['vision'].astype(np.float32)
            vision[vision == -np.inf] = 0
            vision[vision == np.inf] = 0
            vision = np.nan_to_num(vision, 0)
            
            # Normalize each modality
            text = (text - text.mean(axis=0)) / (text.std(axis=0) + 1e-8)
            audio = (audio - audio.mean(axis=0)) / (audio.std(axis=0) + 1e-8)
            vision = (vision - vision.mean(axis=0)) / (vision.std(axis=0) + 1e-8)
            
            self.text = torch.tensor(text)
            self.audio = torch.tensor(audio)
            self.vision = torch.tensor(vision)
            self.label = torch.tensor(dataset['labels'].astype(np.int64), dtype=torch.long)
            
        elif isinstance(dataset, list):
            # Extract text, audio, vision from the dataset
            text = [item[0] for item in dataset]
            audio = [item[1] for item in dataset]
            vision = [item[2] for item in dataset]
            labels = [item[3] if len(item) > 3 else 0 for item in dataset]
            
            def process_modality(data):
                processed = []
                for item in data:
                    try:
                        # Handle nested sequences and convert to float
                        if isinstance(item, (tuple, list)):
                            # Flatten nested structure and convert to float
                            flattened = []
                            for x in item:
                                if isinstance(x, (tuple, list)):
                                    flattened.extend([float(y) if isinstance(y, str) else y for y in x])
                                else:
                                    flattened.append(float(x) if isinstance(x, str) else x)
                            arr = np.array(flattened, dtype=np.float32)
                        else:
                            arr = np.array(item, dtype=np.float32)
                            
                        # Handle inf and nan values
                        arr = np.nan_to_num(arr, 0)
                        arr[arr == -np.inf] = 0
                        arr[arr == np.inf] = 0
                        processed.append(arr)
                    except (ValueError, TypeError) as e:
                        # Use zeros as fallback for invalid data
                        processed.append(np.zeros(300, dtype=np.float32))
                return processed
            
            # Process each modality
            text_arrays = process_modality(text)
            audio_arrays = process_modality(audio)
            vision_arrays = process_modality(vision)
            
            # Get maximum lengths
            max_text_len = max(arr.shape[0] for arr in text_arrays)
            max_audio_len = max(arr.shape[0] for arr in audio_arrays)
            max_vision_len = max(arr.shape[0] for arr in vision_arrays)
            
            # Create padded arrays with correct dimensions
            text_padded = np.zeros((len(text_arrays), max_text_len), dtype=np.float32)
            audio_padded = np.zeros((len(audio_arrays), max_audio_len), dtype=np.float32)
            vision_padded = np.zeros((len(vision_arrays), max_vision_len), dtype=np.float32)
            
            # Fill padded arrays
            for i, arr in enumerate(text_arrays):
                text_padded[i, :arr.shape[0]] = arr
            for i, arr in enumerate(audio_arrays):
                audio_padded[i, :arr.shape[0]] = arr
            for i, arr in enumerate(vision_arrays):
                vision_padded[i, :arr.shape[0]] = arr
            
            # Normalize each modality
            text_padded = (text_padded - text_padded.mean(axis=0)) / (text_padded.std(axis=0) + 1e-8)
            audio_padded = (audio_padded - audio_padded.mean(axis=0)) / (audio_padded.std(axis=0) + 1e-8)
            vision_padded = (vision_padded - vision_padded.mean(axis=0)) / (vision_padded.std(axis=0) + 1e-8)
            
            # Ensure dimensions match model expectations
            if text_padded.shape[1] < 300:  # Pad text to 300 if needed
                padded = np.zeros((text_padded.shape[0], 300), dtype=np.float32)
                padded[:, :text_padded.shape[1]] = text_padded
                text_padded = padded
            elif text_padded.shape[1] > 300:  # Truncate text if too long
                text_padded = text_padded[:, :300]
                
            if audio_padded.shape[1] < 74:  # Pad audio to 74 if needed
                padded = np.zeros((audio_padded.shape[0], 74), dtype=np.float32)
                padded[:, :audio_padded.shape[1]] = audio_padded
                audio_padded = padded
            elif audio_padded.shape[1] > 74:  # Truncate audio if too long
                audio_padded = audio_padded[:, :74]
                
            if vision_padded.shape[1] < 35:  # Pad vision to 35 if needed
                padded = np.zeros((vision_padded.shape[0], 35), dtype=np.float32)
                padded[:, :vision_padded.shape[1]] = vision_padded
                vision_padded = padded
            elif vision_padded.shape[1] > 35:  # Truncate vision if too long
                vision_padded = vision_padded[:, :35]
            
            self.text = torch.tensor(text_padded)
            self.audio = torch.tensor(audio_padded)
            self.vision = torch.tensor(vision_padded)
            self.label = torch.tensor(labels, dtype=torch.long)
        else:
            raise ValueError(f"Unknown dataset format: {type(dataset)}")

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        vision = self.vision[index]
        audio = self.audio[index]
        label = self.label[index]
        return text, audio, vision, label









