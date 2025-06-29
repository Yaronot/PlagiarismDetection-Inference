#!/usr/bin/env python3
"""
Apply Trained Plagiarism Detection Model to Real Articles
Usage: python apply_plagiarism_detection.py --model_path best_siamese_bert.pth --articles_dir /path/to/articles [options]
"""

import argparse
import os
import sys
import json
import re
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from itertools import combinations
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import the model architecture from your original script
class SiameseBERT(nn.Module):
    """Siamese Network with BERT encoder and similarity classifier"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', dropout: float = 0.3):
        super(SiameseBERT, self).__init__()
        
        # Shared BERT encoder
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Similarity classifier
        bert_dim = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(bert_dim * 3, 512),  # concatenated + absolute difference
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        # Get embeddings from both texts using shared BERT
        outputs_1 = self.bert(input_ids_1, attention_mask_1)
        outputs_2 = self.bert(input_ids_2, attention_mask_2)
        
        # Use [CLS] pooled output
        emb_1 = outputs_1.pooler_output
        emb_2 = outputs_2.pooler_output
        
        # Apply dropout
        emb_1 = self.dropout(emb_1)
        emb_2 = self.dropout(emb_2)
        
        # Create feature vector: [emb1, emb2, |emb1-emb2|]
        diff = torch.abs(emb_1 - emb_2)
        combined = torch.cat([emb_1, emb_2, diff], dim=1)
        
        # Predict similarity
        similarity = self.classifier(combined)
        return similarity.squeeze()

class ArticleParagraphExtractor:
    """Extract paragraphs from articles using the existing extract_paragraphs.py script"""
    
    def __init__(self, extract_script_path: str = "extract_paragraphs.py"):
        self.extract_script_path = extract_script_path
        
    def extract_paragraphs_from_file(self, tex_file_path: str) -> list:
        """Extract paragraphs from a single .tex file"""
        try:
            # Run the extraction script
            result = subprocess.run([
                'python3', self.extract_script_path, tex_file_path
            ], capture_output=True, text=True, check=True)
            
            # Parse the output to extract paragraphs
            paragraphs = []
            lines = result.stdout.split('\n')
            
            current_paragraph = ""
            in_paragraph = False
            
            for line in lines:
                if line.startswith("Paragraph ") and "chars)" in line:
                    if current_paragraph.strip():
                        paragraphs.append(current_paragraph.strip())
                    current_paragraph = ""
                    in_paragraph = False
                elif line.startswith("----------------------------------------"):
                    if not in_paragraph:
                        in_paragraph = True
                    else:
                        if current_paragraph.strip():
                            paragraphs.append(current_paragraph.strip())
                        current_paragraph = ""
                        in_paragraph = False
                elif in_paragraph and line.strip():
                    current_paragraph += line + " "
            
            # Add the last paragraph if exists
            if current_paragraph.strip():
                paragraphs.append(current_paragraph.strip())
            
            # Filter out very short paragraphs (less than 50 characters)
            paragraphs = [p for p in paragraphs if len(p) >= 50]
            
            return paragraphs
            
        except subprocess.CalledProcessError as e:
            print(f"Error extracting paragraphs from {tex_file_path}: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error processing {tex_file_path}: {e}")
            return []

class PlagiarismDetector:
    """Main class for detecting plagiarism between articles"""
    
    def __init__(self, model_path: str, model_name: str = 'bert-base-uncased', 
                 device: str = None, max_length: int = 512):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        self.model = SiameseBERT(model_name)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def predict_similarity(self, text1: str, text2: str) -> float:
        """Predict similarity between two texts"""
        # Tokenize texts
        encoding1 = self.tokenizer(
            text1, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        encoding2 = self.tokenizer(
            text2, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        # Move to device
        input_ids_1 = encoding1['input_ids'].to(self.device)
        attention_mask_1 = encoding1['attention_mask'].to(self.device)
        input_ids_2 = encoding2['input_ids'].to(self.device)
        attention_mask_2 = encoding2['attention_mask'].to(self.device)
        
        with torch.no_grad():
            similarity = self.model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
        
        return similarity.item()
    
    def check_token_length(self, text: str) -> bool:
        """Check if text fits within token limit"""
        tokens = self.tokenizer(text, add_special_tokens=True)['input_ids']
        return len(tokens) <= self.max_length
    
    def analyze_articles(self, articles_data: dict, similarity_threshold: float = 0.7) -> list:
        """Analyze multiple articles for plagiarism"""
        results = []
        article_names = list(articles_data.keys())
        
        print(f"Analyzing {len(article_names)} articles...")
        
        # Compare all pairs of articles
        for i, article1 in enumerate(article_names):
            for j, article2 in enumerate(article_names):
                if i >= j:  # Avoid duplicates and self-comparison
                    continue
                
                print(f"\nComparing {article1} vs {article2}")
                
                paragraphs1 = articles_data[article1]
                paragraphs2 = articles_data[article2]
                
                # Filter paragraphs by token length
                valid_paragraphs1 = [p for p in paragraphs1 if self.check_token_length(p)]
                valid_paragraphs2 = [p for p in paragraphs2 if self.check_token_length(p)]
                
                print(f"  Article 1: {len(valid_paragraphs1)}/{len(paragraphs1)} valid paragraphs")
                print(f"  Article 2: {len(valid_paragraphs2)}/{len(paragraphs2)} valid paragraphs")
                
                # Compare all paragraph combinations
                for p1_idx, paragraph1 in enumerate(tqdm(valid_paragraphs1, desc="Comparing paragraphs")):
                    for p2_idx, paragraph2 in enumerate(valid_paragraphs2):
                        similarity = self.predict_similarity(paragraph1, paragraph2)
                        
                        if similarity >= similarity_threshold:
                            results.append({
                                'article1': article1,
                                'article2': article2,
                                'paragraph1_idx': p1_idx,
                                'paragraph2_idx': p2_idx,
                                'similarity_score': similarity,
                                'paragraph1': paragraph1,
                                'paragraph2': paragraph2,
                                'paragraph1_length': len(paragraph1),
                                'paragraph2_length': len(paragraph2)
                            })
        
        return results
    
    def analyze_single_article(self, article_name: str, paragraphs: list, 
                             similarity_threshold: float = 0.7) -> list:
        """Analyze a single article for internal plagiarism"""
        results = []
        
        # Filter paragraphs by token length
        valid_paragraphs = [(i, p) for i, p in enumerate(paragraphs) if self.check_token_length(p)]
        
        print(f"Analyzing {article_name} for internal plagiarism...")
        print(f"Valid paragraphs: {len(valid_paragraphs)}/{len(paragraphs)}")
        
        # Compare all paragraph combinations within the article
        for i, (idx1, paragraph1) in enumerate(tqdm(valid_paragraphs, desc="Comparing paragraphs")):
            for j, (idx2, paragraph2) in enumerate(valid_paragraphs):
                if i >= j:  # Avoid duplicates and self-comparison
                    continue
                
                similarity = self.predict_similarity(paragraph1, paragraph2)
                
                if similarity >= similarity_threshold:
                    results.append({
                        'article': article_name,
                        'paragraph1_idx': idx1,
                        'paragraph2_idx': idx2,
                        'similarity_score': similarity,
                        'paragraph1': paragraph1,
                        'paragraph2': paragraph2,
                        'paragraph1_length': len(paragraph1),
                        'paragraph2_length': len(paragraph2)
                    })
        
        return results

def find_tex_files(directory: str, limit: int = None) -> list:
    """Find all .tex files in directory"""
    tex_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.tex'):
                tex_files.append(os.path.join(root, file))
                if limit and len(tex_files) >= limit:
                    return tex_files[:limit]
    return tex_files

def save_results(results: list, output_file: str):
    """Save results to JSON and CSV files"""
    # Save as JSON for full data
    json_file = output_file.replace('.csv', '.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save as CSV for easy viewing
    if results:
        df_results = []
        for result in results:
            row = {
                'article1': result.get('article1', result.get('article', '')),
                'article2': result.get('article2', ''),
                'paragraph1_idx': result['paragraph1_idx'],
                'paragraph2_idx': result['paragraph2_idx'],
                'similarity_score': result['similarity_score'],
                'paragraph1_length': result['paragraph1_length'],
                'paragraph2_length': result['paragraph2_length'],
                'paragraph1_preview': result['paragraph1'][:100] + '...' if len(result['paragraph1']) > 100 else result['paragraph1'],
                'paragraph2_preview': result['paragraph2'][:100] + '...' if len(result['paragraph2']) > 100 else result['paragraph2']
            }
            df_results.append(row)
        
        df = pd.DataFrame(df_results)
        df.to_csv(output_file, index=False)
        
        print(f"Results saved to:")
        print(f"  - JSON: {json_file}")
        print(f"  - CSV: {output_file}")
    else:
        print("No plagiarism instances found.")

def main():
    parser = argparse.ArgumentParser(description='Apply trained plagiarism detection model to real articles')
    parser.add_argument('--model_path', required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--articles_dir', required=True, help='Directory containing .tex articles')
    parser.add_argument('--extract_script', default='extract_paragraphs.py', help='Path to paragraph extraction script')
    parser.add_argument('--similarity_threshold', type=float, default=0.7, help='Similarity threshold for flagging plagiarism')
    parser.add_argument('--max_articles', type=int, default=2, help='Maximum number of articles to analyze')
    parser.add_argument('--output_file', default='plagiarism_results.csv', help='Output file for results')
    parser.add_argument('--mode', choices=['cross', 'internal', 'both'], default='cross', 
                       help='Analysis mode: cross (between articles), internal (within articles), or both')
    parser.add_argument('--model_name', default='bert-base-uncased', help='BERT model name used for training')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist")
        sys.exit(1)
    
    if not os.path.exists(args.articles_dir):
        print(f"Error: Articles directory {args.articles_dir} does not exist")
        sys.exit(1)
    
    if not os.path.exists(args.extract_script):
        print(f"Error: Extract script {args.extract_script} does not exist")
        sys.exit(1)
    
    print("=== Plagiarism Detection Analysis ===")
    print(f"Model: {args.model_path}")
    print(f"Articles directory: {args.articles_dir}")
    print(f"Similarity threshold: {args.similarity_threshold}")
    print(f"Max articles: {args.max_articles}")
    print(f"Analysis mode: {args.mode}")
    
    # Find .tex files
    tex_files = find_tex_files(args.articles_dir, limit=args.max_articles)
    
    if not tex_files:
        print(f"No .tex files found in {args.articles_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(tex_files)} .tex files")
    
    # Extract paragraphs from all articles
    extractor = ArticleParagraphExtractor(args.extract_script)
    articles_data = {}
    
    print("\nExtracting paragraphs from articles...")
    for tex_file in tqdm(tex_files):
        article_name = os.path.basename(tex_file)
        paragraphs = extractor.extract_paragraphs_from_file(tex_file)
        if paragraphs:
            articles_data[article_name] = paragraphs
            print(f"  {article_name}: {len(paragraphs)} paragraphs")
        else:
            print(f"  {article_name}: No paragraphs extracted")
    
    if not articles_data:
        print("No paragraphs extracted from any articles")
        sys.exit(1)
    
    # Initialize plagiarism detector
    detector = PlagiarismDetector(args.model_path, model_name=args.model_name)
    
    all_results = []
    
    # Cross-article analysis
    if args.mode in ['cross', 'both']:
        if len(articles_data) >= 2:
            cross_results = detector.analyze_articles(articles_data, args.similarity_threshold)
            all_results.extend(cross_results)
            print(f"\nCross-article analysis: {len(cross_results)} potential plagiarism instances found")
        else:
            print("\nSkipping cross-article analysis (need at least 2 articles)")
    
    # Internal article analysis
    if args.mode in ['internal', 'both']:
        for article_name, paragraphs in articles_data.items():
            internal_results = detector.analyze_single_article(article_name, paragraphs, args.similarity_threshold)
            all_results.extend(internal_results)
            print(f"Internal analysis for {article_name}: {len(internal_results)} potential instances found")
    
    # Save results
    save_results(all_results, args.output_file)
    
    # Summary
    print(f"\n=== Analysis Complete ===")
    print(f"Total potential plagiarism instances: {len(all_results)}")
    if all_results:
        similarities = [r['similarity_score'] for r in all_results]
        print(f"Similarity scores range: {min(similarities):.3f} - {max(similarities):.3f}")
        print(f"Average similarity: {np.mean(similarities):.3f}")

if __name__ == "__main__":
    main()
