#!/usr/bin/env python3
"""
PDF Processing Script - Works for both Round 1A and Round 1B
- Round 1A: Extracts structured information (title, headings, description)
- Round 1B: Given a persona + job-to-be-done, extracts top relevant sections
"""

import os
import json
import fitz  # PyMuPDF
import re
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
import time
from datetime import datetime


class PDFProcessor:
    def __init__(self):
        """Initialize the PDF processor with the offline model."""
        print("🔄 Loading AI model...")
        
        # Import here to avoid any issues
        from sentence_transformers import SentenceTransformer
        
        # Let's debug what's actually in the models directory
        print("📁 Checking models directory...")
        if os.path.exists('/app/models'):
            print("✅ /app/models directory exists")
            for root, dirs, files in os.walk('/app/models'):
                print(f"📂 Directory: {root}")
                if files:
                    print(f"📄 Files: {files[:10]}")  # Show first 10 files
                if dirs:
                    print(f"📁 Subdirs: {dirs[:10]}")  # Show first 10 subdirs
        else:
            print("❌ /app/models directory does not exist!")
            raise Exception("Models directory not found!")
        
        # Try multiple approaches to find and load the model
        model_loaded = False
        
        print("🔍 Approach 1: Searching for model directory...")
        for root, dirs, files in os.walk('/app/models'):
            if 'config.json' in files:
                print(f"🎯 Found config.json in: {root}")
                if any('.bin' in f or '.safetensors' in f for f in files):
                    print(f"🎯 Found model files in: {root}")
                    try:
                        self.model = SentenceTransformer(root)
                        print("✅ Model loaded successfully from auto-detected path!")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"❌ Failed to load from {root}: {e}")
        
        if not model_loaded:
            print("🔍 Approach 2: Trying common paths...")
            common_paths = [
                '/app/models/sentence-transformers_all-MiniLM-L6-v2',
                '/app/models/models--sentence-transformers--all-MiniLM-L6-v2',
            ]
            for path in common_paths:
                print(f"🔍 Checking path: {path}")
                if os.path.exists(path):
                    print(f"✅ Path exists: {path}")
                    try:
                        self.model = SentenceTransformer(path)
                        print("✅ Model loaded successfully from common path!")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"❌ Failed to load from {path}: {e}")
                else:
                    print(f"❌ Path does not exist: {path}")
        
        if not model_loaded:
            print("🔍 Approach 3: Trying to load from cache...")
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/app/models')
                print("✅ Model loaded successfully from cache!")
                model_loaded = True
            except Exception as e:
                print(f"❌ Failed to load from cache: {e}")
        
        if not model_loaded:
            print("❌ Could not load the AI model!")
            print("🔧 Using fallback mode - will work without AI clustering")
            self.model = None
        else:
            print("🎉 Model initialization complete!")

    def extract_text_with_structure(self, pdf_path):
        doc = fitz.open(pdf_path)
        text_blocks = []
        all_text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        font_sizes = []
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                line_text += text + " "
                                font_sizes.append(span["size"])
                        if line_text.strip():
                            avg_font_size = np.mean(font_sizes) if font_sizes else 12
                            text_blocks.append({
                                'text': line_text.strip(),
                                'font_size': avg_font_size,
                                'page': page_num + 1
                            })
                            all_text += line_text + " "
        doc.close()
        return text_blocks, all_text

    def identify_title_and_headings(self, text_blocks):
        if not text_blocks:
            return None, []

        font_sizes = [block['font_size'] for block in text_blocks]
        avg_font_size = np.mean(font_sizes)
        max_font_size = max(font_sizes)

        title_candidates = [block for block in text_blocks[:10] if block['font_size'] >= max_font_size * 0.9 and len(block['text']) > 10]
        title = max(title_candidates, key=lambda x: len(x['text']))['text'] if title_candidates else None

        heading_threshold = avg_font_size * 1.2
        heading_candidates = []
        for block in text_blocks:
            text = block['text']
            if block['font_size'] >= heading_threshold and len(text) < 200 and text != title:
                if (text.isupper() or re.match(r'^[A-Z][a-z].*[^.]$', text) or re.match(r'^\d+\.\s+[A-Z]', text)):
                    heading_candidates.append({
                        'text': text,
                        'font_size': block['font_size'],
                        'page': block['page']
                    })

        heading_candidates.sort(key=lambda x: x['font_size'], reverse=True)
        unique_sizes = sorted(set(h['font_size'] for h in heading_candidates), reverse=True)

        headings = []
        for heading in heading_candidates:
            size_rank = unique_sizes.index(heading['font_size'])
            level = f"H{min(size_rank + 1, 3)}"
            if heading['text'] not in [h['text'] for h in headings]:
                headings.append({
                    'level': level,
                    'text': heading['text'],
                    'page': heading['page']
                })

        return title, headings[:10]

    def generate_description(self, text, title, outline):
        clean_text = text
        if title:
            clean_text = clean_text.replace(title, "")
        for heading in outline:
            clean_text = clean_text.replace(heading['text'], "")

        sentences = re.split(r'[.!?]+', clean_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return "No descriptive content found."
        if len(sentences) <= 3:
            return " ".join(sentences[:3]) + "."

        if self.model:
            try:
                embeddings = self.model.encode(sentences)
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(embeddings)
                result = []
                for i in range(3):
                    indices = [j for j in range(len(clusters)) if clusters[j] == i]
                    best_idx = min(indices, key=lambda idx: np.linalg.norm(embeddings[idx] - kmeans.cluster_centers_[i]))
                    result.append(sentences[best_idx])
                return " ".join(result) + "."
            except:
                return " ".join(sentences[:3]) + "."
        else:
            return " ".join(sentences[:3]) + "."

    def process_pdf(self, pdf_path, output_path):
        text_blocks, full_text = self.extract_text_with_structure(pdf_path)
        if not text_blocks:
            result = {"title": "Empty document", "outline": [], "description": "No content found."}
        else:
            title, outline = self.identify_title_and_headings(text_blocks)
            description = self.generate_description(full_text, title, outline)
            result = {
                "title": title or "Untitled Document",
                "outline": outline,
                "description": description
            }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ Generated: {output_path}")


def process_round_1b(processor, pdf_files, output_dir, persona, job):
    query = f"{persona}. {job}"
    if not processor.model:
        print("⚠️ No model found, cannot run Round 1B")
        return

    query_embedding = processor.model.encode([query])
    section_scores = []
    for pdf_file in pdf_files:
        blocks, _ = processor.extract_text_with_structure(str(pdf_file))
        for block in blocks:
            try:
                emb = processor.model.encode([block['text']])
                score = cosine_similarity(query_embedding, emb)[0][0]
                section_scores.append({
                    "document": pdf_file.name,
                    "page": block["page"],
                    "section_title": block["text"][:100],
                    "refined_text": block["text"],
                    "score": float(score)
                })
            except:
                continue

    top_sections = sorted(section_scores, key=lambda x: x['score'], reverse=True)[:10]
    for i, sec in enumerate(top_sections):
        sec["importance_rank"] = i + 1

    output = {
        "metadata": {
            "documents": [f.name for f in pdf_files],
            "persona": persona,
            "job": job,
            "timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [
            {
                "document": s["document"],
                "page": s["page"],
                "section_title": s["section_title"],
                "importance_rank": s["importance_rank"]
            } for s in top_sections
        ],
        "subsection_analysis": [
            {
                "document": s["document"],
                "page": s["page"],
                "refined_text": s["refined_text"],
                "importance_rank": s["importance_rank"]
            } for s in top_sections
        ]
    }

    with open(output_dir / "output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("✅ Round 1B output saved as output.json")


def main():
    print("\n🚀 Starting PDF processor...")
    start = time.time()

    processor = PDFProcessor()
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    output_dir.mkdir(exist_ok=True)

    pdf_files = list(input_dir.glob("*.pdf"))
    task_file = input_dir / "task.json"

    if not pdf_files:
        print("⚠️ No PDF files in input folder.")
        return

    if task_file.exists():
        with open(task_file) as f:
            task = json.load(f)
        process_round_1b(
            processor,
            pdf_files,
            output_dir,
            task.get("persona", ""),
            task.get("job", "")
        )
    else:
        for pdf_file in pdf_files:
            output_file = output_dir / f"{pdf_file.stem}.json"
            processor.process_pdf(str(pdf_file), str(output_file))

    print(f"\n⏱️ Done in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
