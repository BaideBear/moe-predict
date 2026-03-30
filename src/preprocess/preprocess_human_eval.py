import json
import os
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from datasets import load_dataset


class HumanEvalPreprocessor:
    def __init__(
        self,
        output_path: str = "/data1/gx/MoE-predict/dataset/processed",
        train_ratio: float = 0.9
    ):
        self.output_path = output_path
        self.train_ratio = train_ratio
        
        os.makedirs(os.path.join(output_path, "train"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "test"), exist_ok=True)
    
    def load_human_eval_dataset(self) -> List[Dict]:
        all_data = []
        
        print("Loading HumanEval dataset from Hugging Face...")
        
        try:
            dataset = load_dataset("openai_humaneval")
            
            if "test" in dataset:
                for item in tqdm(dataset["test"], desc="Loading HumanEval"):
                    sample = {
                        "task_id": item["task_id"],
                        "prompt": item["prompt"],
                        "canonical_solution": item["canonical_solution"],
                        "test": item["test"],
                        "entry_point": item["entry_point"]
                    }
                    all_data.append(sample)
            
        except Exception as e:
            print(f"Error loading HumanEval dataset: {e}")
            return []
        
        print(f"Total loaded: {len(all_data)} samples")
        return all_data
    
    def format_sample(self, sample: Dict[str, Any]) -> str:
        prompt = sample["prompt"]
        canonical_solution = sample["canonical_solution"]
        
        text = f"Problem: {prompt}\n\nSolution: {canonical_solution}"
        
        return text
    
    def process_dataset(self, data: List[Dict]) -> List[Dict]:
        processed_data = []
        
        for sample in tqdm(data, desc="Processing samples"):
            text = self.format_sample(sample)
            
            processed_sample = {
                "text": text,
                "metadata": {
                    "source": "human_eval",
                    "category": "code_generation",
                    "task_id": sample["task_id"],
                    "entry_point": sample["entry_point"]
                }
            }
            
            processed_data.append(processed_sample)
        
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
        print("HumanEval Dataset Preprocessing")
        print("=" * 60)
        
        print("\nStep 1: Loading HumanEval dataset...")
        raw_data = self.load_human_eval_dataset()
        
        if len(raw_data) == 0:
            print("Error: No data loaded. Exiting.")
            return
        
        print("\nStep 2: Processing dataset...")
        processed_data = self.process_dataset(raw_data)
        
        print("\nStep 3: Splitting dataset...")
        train_data, test_data = self.split_dataset(processed_data)
        
        print("\nStep 4: Saving processed data...")
        train_file = os.path.join(self.output_path, "train", "human_eval.jsonl")
        test_file = os.path.join(self.output_path, "test", "human_eval.jsonl")
        
        self.save_data(train_data, train_file)
        self.save_data(test_data, test_file)
        
        print("\n" + "=" * 60)
        print("Preprocessing completed successfully!")
        print(f"Train set: {train_file}")
        print(f"Test set: {test_file}")
        print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess HumanEval dataset')
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
    
    args = parser.parse_args()
    
    preprocessor = HumanEvalPreprocessor(
        output_path=args.output_path,
        train_ratio=args.train_ratio
    )
    
    preprocessor.run()


if __name__ == "__main__":
    main()
