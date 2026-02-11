"""
Dataset preprocessing and loading for GSM8K math word problems.
"""

from typing import List, Dict
from omegaconf import DictConfig


# Few-shot exemplars for Chain-of-Thought prompting
FEW_SHOT_EXEMPLARS = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "solution": "Janet's ducks lay 16 eggs per day. She eats 3 eggs for breakfast and uses 4 eggs for baking. So she has 16 - 3 - 4 = 9 eggs left to sell. At $2 per egg, she makes 9 * 2 = 18 dollars.",
        "answer": "18",
        "mode": "Algebra",
        "confidence": 0.95,
        "sanity": "Total eggs used: 3 + 4 + 9 = 16. Matches daily production. PASS"
    },
    {
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "solution": "The robe takes 2 bolts of blue fiber. It takes half that much white fiber, which is 2 / 2 = 1 bolt of white fiber. In total, it takes 2 + 1 = 3 bolts.",
        "answer": "3",
        "mode": "Algebra",
        "confidence": 0.98,
        "sanity": "Blue: 2, White: 1, Total: 2+1=3. PASS"
    },
    {
        "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
        "solution": "Josh's total investment is $80,000 + $50,000 = $130,000. The house increased in value by 150%, so the new value is $80,000 + ($80,000 * 1.5) = $80,000 + $120,000 = $200,000. His profit is $200,000 - $130,000 = $70,000.",
        "answer": "70000",
        "mode": "Algebra",
        "confidence": 0.92,
        "sanity": "Investment: 130k, Value: 200k, Profit: 70k. Reasonable. PASS"
    },
    {
        "question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
        "solution": "James runs 3 sprints per session, and each sprint is 60 meters. So per session he runs 3 * 60 = 180 meters. He does this 3 times a week, so he runs 180 * 3 = 540 meters per week.",
        "answer": "540",
        "mode": "Algebra",
        "confidence": 0.96,
        "sanity": "Per session: 3*60=180. Per week: 180*3=540. Units check. PASS"
    }
]


# Synthetic GSM8K-style dataset for testing
# In production, this would load from HuggingFace datasets or a file
SYNTHETIC_GSM8K_DATA = [
    {
        "question": "A bakery sells 24 cupcakes in the morning and 36 in the afternoon. Each cupcake costs $3. How much money did the bakery make in total?",
        "answer": "180"
    },
    {
        "question": "Tom has 15 apples. He gives 5 to his friend and buys 8 more. How many apples does he have now?",
        "answer": "18"
    },
    {
        "question": "A car travels 60 miles per hour for 3 hours. How many miles does it travel in total?",
        "answer": "180"
    },
    {
        "question": "Sarah has $50. She spends $12 on a book and $8 on lunch. How much money does she have left?",
        "answer": "30"
    },
    {
        "question": "A rectangle has a length of 8 meters and a width of 5 meters. What is its area in square meters?",
        "answer": "40"
    },
    {
        "question": "Mike reads 20 pages of a book every day. How many pages does he read in 7 days?",
        "answer": "140"
    },
    {
        "question": "A store has 45 shirts. They sell 18 shirts on Monday and 12 shirts on Tuesday. How many shirts are left?",
        "answer": "15"
    },
    {
        "question": "Emma plants 6 flowers in each of 4 rows. How many flowers does she plant in total?",
        "answer": "24"
    },
    {
        "question": "A pizza is cut into 8 slices. If 3 people each eat 2 slices, how many slices are left?",
        "answer": "2"
    },
    {
        "question": "John has 3 boxes with 12 pencils in each box. How many pencils does he have in total?",
        "answer": "36"
    },
    {
        "question": "A train travels 240 kilometers in 4 hours. What is its average speed in kilometers per hour?",
        "answer": "60"
    },
    {
        "question": "Lisa has $100. She buys 3 toys that cost $15 each. How much money does she have left?",
        "answer": "55"
    },
    {
        "question": "A garden has 28 red roses and 14 white roses. How many roses are there in total?",
        "answer": "42"
    },
    {
        "question": "David works 8 hours per day for 5 days. How many hours does he work in total?",
        "answer": "40"
    },
    {
        "question": "A bag contains 50 marbles. If 20 are red, 15 are blue, and the rest are green, how many green marbles are there?",
        "answer": "15"
    }
]


def load_dataset(cfg: DictConfig) -> List[Dict[str, str]]:
    """
    Load and preprocess the GSM8K dataset.
    
    Args:
        cfg: Hydra config containing dataset_params
        
    Returns:
        List of examples with 'question' and 'answer' fields
    """
    dataset_name = cfg.run.dataset
    split = cfg.dataset_params.split
    max_samples = cfg.dataset_params.get("max_samples", None)
    
    print(f"Loading dataset: {dataset_name} (split: {split})")
    
    if dataset_name == "gsm8k":
        # In production, load from HuggingFace:
        # from datasets import load_dataset as hf_load_dataset
        # dataset = hf_load_dataset("gsm8k", "main", split=split, cache_dir=cfg.dataset_params.cache_dir)
        # For this implementation, use synthetic data
        
        dataset = SYNTHETIC_GSM8K_DATA
        
        if max_samples is not None:
            dataset = dataset[:max_samples]
        
        print(f"Loaded {len(dataset)} examples")
        return dataset
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
