import json
import os
from typing import Dict, List, Any, Optional
from tqdm import tqdm
from datasets import load_dataset


class MMLUPreprocessor:
    def __init__(
        self,
        output_path: str = "/data1/gx/MoE-predict/dataset/processed",
        train_ratio: float = 0.9,
        subjects: Optional[List[str]] = None
    ):
        self.output_path = output_path
        self.train_ratio = train_ratio
        self.subjects = subjects
        
        os.makedirs(os.path.join(output_path, "train"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "test"), exist_ok=True)
    
    def load_mmlu_dataset(self) -> List[Dict]:
        all_data = []
        
        print("Loading MMLU dataset from Hugging Face...")
        
        if self.subjects and len(self.subjects) > 0:
            subjects_to_load = self.subjects
        else:
            subjects_to_load = [
                "abstract_algebra", "anatomy", "astronomy", "business_ethics",
                "clinical_knowledge", "college_biology", "college_chemistry",
                "college_computer_science", "college_mathematics", "college_medicine",
                "college_physics", "computer_security", "conceptual_physics",
                "econometrics", "electrical_engineering", "elementary_mathematics",
                "formal_logic", "global_facts", "high_school_biology",
                "high_school_chemistry", "high_school_computer_science",
                "high_school_european_history", "high_school_geography",
                "high_school_government_and_politics", "high_school_macroeconomics",
                "high_school_mathematics", "high_school_microeconomics",
                "high_school_physics", "high_school_psychology", "high_school_statistics",
                "high_school_us_history", "high_school_world_history", "human_aging",
                "human_sexuality", "international_law", "jurisprudence",
                "logical_fallacies", "machine_learning", "management", "marketing",
                "medical_genetics", "miscellaneous", "moral_disputes",
                "moral_scenarios", "nutrition", "philosophy", "prehistory",
                "professional_accounting", "professional_law", "professional_medicine",
                "professional_psychology", "public_relations", "security_studies",
                "sociology", "us_foreign_policy", "virology", "world_religions"
            ]
        
        for subject in tqdm(subjects_to_load, desc="Loading subjects"):
            try:
                dataset = load_dataset("cais/mmlu", subject)
                
                for split_name in ["test", "validation", "dev"]:
                    if split_name in dataset:
                        for item in dataset[split_name]:
                            sample = {
                                "question": item["question"],
                                "choices": item["choices"],
                                "answer": item["answer"],
                                "subject": subject
                            }
                            all_data.append(sample)
                
            except Exception as e:
                print(f"Warning: Failed to load subject '{subject}': {e}")
                continue
        
        print(f"Total loaded: {len(all_data)} samples from {len(subjects_to_load)} subjects")
        return all_data
    
    def format_sample(self, sample: Dict[str, Any]) -> str:
        question = sample["question"]
        choices = sample["choices"]
        
        text = f"Question: {question}\n"
        for idx, choice in enumerate(choices):
            text += f"{chr(65 + idx)}. {choice}\n"
        text += "Answer:"
        
        return text
    
    def process_dataset(self, data: List[Dict]) -> tuple:
        processed_data = []
        
        for sample in tqdm(data, desc="Processing samples"):
            text = self.format_sample(sample)
            
            processed_sample = {
                "text": text,
                "metadata": {
                    "source": "mmlu",
                    "category": sample["subject"],
                    "answer": sample["answer"]
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
        print("MMLU Dataset Preprocessing")
        print("=" * 60)
        
        print("\nStep 1: Loading MMLU dataset...")
        raw_data = self.load_mmlu_dataset()
        
        print("\nStep 2: Processing dataset...")
        processed_data = self.process_dataset(raw_data)
        
        print("\nStep 3: Splitting dataset...")
        train_data, test_data = self.split_dataset(processed_data)
        
        print("\nStep 4: Saving processed data...")
        train_file = os.path.join(self.output_path, "train", "mmlu.jsonl")
        test_file = os.path.join(self.output_path, "test", "mmlu.jsonl")
        
        self.save_data(train_data, train_file)
        self.save_data(test_data, test_file)
        
        print("\n" + "=" * 60)
        print("Preprocessing completed successfully!")
        print(f"Train set: {train_file}")
        print(f"Test set: {test_file}")
        print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess MMLU dataset')
    parser.add_argument(
        '--subjects',
        type=str,
        nargs='+',
        help='List of subjects to load (default: all subjects)'
    )
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
    
    preprocessor = MMLUPreprocessor(
        output_path=args.output_path,
        train_ratio=args.train_ratio,
        subjects=args.subjects
    )
    
    preprocessor.run()


if __name__ == "__main__":
    main()
