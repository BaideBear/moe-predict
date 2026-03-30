import json
import os
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from datasets import load_dataset


class WikiTextPreprocessor:
    def __init__(
        self,
        output_path: str = "/data1/gx/MoE-predict/dataset/processed",
        train_ratio: float = 0.9,
        max_samples: int = 10000,
        text_length: int = 512
    ):
        self.output_path = output_path
        self.train_ratio = train_ratio
        self.max_samples = max_samples
        self.text_length = text_length
        
        os.makedirs(os.path.join(output_path, "train"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "test"), exist_ok=True)
    
    def load_wikitext_dataset(self) -> List[str]:
        all_texts = []
        
        print("Loading WikiText-2 dataset from Hugging Face...")
        
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
            
            for split_name in ["train", "test", "validation"]:
                if split_name in dataset:
                    print(f"Loading {split_name} split...")
                    for item in dataset[split_name]:
                        text = item["text"].strip()
                        if text:
                            all_texts.append(text)
            
        except Exception as e:
            print(f"Error loading WikiText dataset: {e}")
            return []
        
        print(f"Total loaded: {len(all_texts)} texts")
        return all_texts
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.text_length):
            chunk = " ".join(words[i:i + self.text_length])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def process_dataset(self, texts: List[str]) -> List[Dict]:
        processed_data = []
        
        print(f"Processing texts and splitting into chunks of {self.text_length} words...")
        
        for text in tqdm(texts, desc="Processing texts"):
            chunks = self.split_text_into_chunks(text)
            
            for chunk in chunks:
                processed_sample = {
                    "text": chunk,
                    "metadata": {
                        "source": "wikitext",
                        "category": "language_modeling"
                    }
                }
                
                processed_data.append(processed_sample)
                
                if len(processed_data) >= self.max_samples:
                    print(f"Reached maximum sample limit: {self.max_samples}")
                    return processed_data
        
        return processed_data
    
    def split_dataset(self, data: List[Dict]) -> tuple:
        import random
        random.shuffle(data)
        
        split_idx = int(len(data) * self.train_ratio)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        print(f"Split dataset: {len(train_data)} train, {len(test_data)} test")
        return train_data, test_data
    
    def save_data(self, data: List[Dict], output_file: str):
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Saved {len(data)} samples to {output_file}")
    
    def run(self):
        print("=" * 60)
        print("WikiText Dataset Preprocessing")
        print("=" * 60)
        
        print("\nStep 1: Loading WikiText dataset...")
        raw_texts = self.load_wikitext_dataset()
        
        if len(raw_texts) == 0:
            print("Error: No data loaded. Exiting.")
            return
        
        print("\nStep 2: Processing dataset...")
        processed_data = self.process_dataset(raw_texts)
        
        print("\nStep 3: Splitting dataset...")
        train_data, test_data = self.split_dataset(processed_data)
        
        print("\nStep 4: Saving processed data...")
        train_file = os.path.join(self.output_path, "train", "wikitext.jsonl")
        test_file = os.path.join(self.output_path, "test", "wikitext.jsonl")
        
        self.save_data(train_data, train_file)
        self.save_data(test_data, test_file)
        
        print("\n" + "=" * 60)
        print("Preprocessing completed successfully!")
        print(f"Train set: {train_file}")
        print(f"Test set: {test_file}")
        print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess WikiText dataset')
    parser.add_argument(
        '--output-path',
        type=str,
        default='/data1/gx/MoE-predict/dataset/processed',
        help='Path to output directory'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.9,
        help='Ratio of training data (default: 0.9)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=10000,
        help='Maximum number of samples to generate (default: 10000)'
    )
    parser.add_argument(
        '--text-length',
        type=int,
        default=512,
        help='Number of words per text chunk (default: 512)'
    )
    
    args = parser.parse_args()
    
    preprocessor = WikiTextPreprocessor(
        output_path=args.output_path,
        train_ratio=args.train_ratio,
        max_samples=args.max_samples,
        text_length=args.text_length
    )
    
    preprocessor.run()


if __name__ == "__main__":
    main()
