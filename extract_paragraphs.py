#!/usr/bin/env python3
"""
Simple LaTeX paragraph extractor for plagiarism detection
Test script for processing TEX files and extracting clean paragraphs
"""

import re
import sys

def clean_latex_text(text):
    """Remove LaTeX commands and clean text for paragraph extraction"""
    
    # Remove comments (lines starting with %)
    text = re.sub(r'^%.*$', '', text, flags=re.MULTILINE)
    
    # Remove common LaTeX commands with arguments
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    
    # Remove common commands without arguments
    latex_commands = [
        r'\\noindent', r'\\newline', r'\\\\', r'\\clearpage', r'\\newpage',
        r'\\frontmatter', r'\\begin\{[^}]*\}', r'\\end\{[^}]*\}',
        r'\\documentclass\[.*?\]\{.*?\}', r'\\usepackage\{.*?\}',
        r'\\pagestyle\{.*?\}', r'\\journal\{.*?\}', r'\\address\{.*?\}'
    ]
    
    for cmd in latex_commands:
        text = re.sub(cmd, '', text)
    
    # Remove math environments (between $ or \[ \])
    text = re.sub(r'\$\$.*?\$\$', '[MATH]', text, flags=re.DOTALL)
    text = re.sub(r'\$.*?\$', '[MATH]', text)
    text = re.sub(r'\\begin\{equation\}.*?\\end\{equation\}', '[EQUATION]', text, flags=re.DOTALL)
    
    # Remove section markers but keep the content
    text = re.sub(r'\\section\*?\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\subsection\*?\{([^}]*)\}', r'\1', text)
    
    # Remove bibliography entries
    text = re.sub(r'\\cite\{[^}]*\}', '[CITATION]', text)
    
    # Remove labels and refs
    text = re.sub(r'\\label\{[^}]*\}', '', text)
    text = re.sub(r'\\ref\{[^}]*\}', '[REF]', text)
    
    # Clean up remaining backslashes and braces
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'[{}]', '', text)
    
    return text

def extract_paragraphs(tex_content, min_length=50):
    """Extract meaningful paragraphs from LaTeX content"""
    
    # Clean the LaTeX
    clean_text = clean_latex_text(tex_content)
    
    # Split into potential paragraphs (double newlines or more)
    paragraphs = re.split(r'\n\s*\n+', clean_text)
    
    # Clean and filter paragraphs
    clean_paragraphs = []
    for para in paragraphs:
        # Remove extra whitespace and newlines
        para = ' '.join(para.split())
        
        # Skip if too short or mostly empty
        if len(para) < min_length:
            continue
            
        # Skip if mostly mathematical notation or commands
        if para.count('[') > len(para) // 10:  # Too many math/ref markers
            continue
            
        # Skip headers and short titles
        if len(para) < 100 and any(word in para.lower() for word in ['abstract', 'introduction', 'conclusion', 'references']):
            continue
            
        clean_paragraphs.append(para.strip())
    
    return clean_paragraphs

def main():
    # Read the TEX file (you'll need to provide the path)
    tex_file = "/sci/labs/orzuk/orzuk/teaching/big_data_project_52017/2024_25/arxiv_data/full_papers/2010/01/01/tex/1001.0267v1.tex"    
    try:
        with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
            tex_content = f.read()
    except FileNotFoundError:
        print(f"Error: File {tex_file} not found")
        print("Please update the file path in the script")
        return
    
    # Extract paragraphs
    paragraphs = extract_paragraphs(tex_content, min_length=50)
    
    print(f"Extracted {len(paragraphs)} paragraphs from {tex_file}")
    print("=" * 60)
    
    # Display first few paragraphs as examples
    for i, para in enumerate(paragraphs,  1):
        print(f"\nParagraph {i} ({len(para)} chars):")
        print("-" * 40)
        print(para)
        print("-" * 40)
    
    
    # Show statistics
    lengths = [len(p) for p in paragraphs]
    if lengths:
        print(f"\nStatistics:")
        print(f"Average paragraph length: {sum(lengths) / len(lengths):.1f} characters")
        print(f"Shortest: {min(lengths)} characters")
        print(f"Longest: {max(lengths)} characters")

if __name__ == "__main__":
    main()
