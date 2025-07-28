# 📄 PDF Processor - Round 1B

This script processes a set of PDFs to extract the most relevant sections based on a **given persona and job-to-be-done**. It uses sentence embeddings and cosine similarity to rank and extract sections that best match the query.

---

## 🎯 Objective

Given:
- A **persona** (e.g., `"Startup Founder"`)
- A **job-to-be-done** (e.g., `"Exploring AI for automating hiring"`)

The script:
1. Embeds the persona + job into a semantic vector.
2. Embeds all text blocks from PDFs.
3. Computes cosine similarity between the query and each block.
4. Returns the **top 10 most relevant sections**, along with metadata.

---

## 📂 Directory Structure

```
/app
 ├── input/
 │   ├── *.pdf           # PDF files to analyze
 │   └── task.json       # Task JSON containing persona and job
 ├── output/
 │   └── output.json     # Final extracted top sections
 ├── models/             # (Optional) Pre-downloaded SentenceTransformer model
 └── processor.py        # Main script
```

---

## 📥 Input: `task.json`

Place this file in the `/app/input` directory:

```json
{
  "persona": "Startup Founder",
  "job": "Exploring AI for automating hiring"
}
```

---

## 🧠 How It Works

- Loads the `SentenceTransformer` model from `/app/models`, or falls back to downloading.
- Embeds the query (`persona + job`) and all sections from PDFs.
- Calculates cosine similarity scores.
- Selects and ranks the **top 10** most relevant blocks.
- Outputs a structured JSON file in `/app/output/output.json`.

---

## 🧾 Output Format: `output/output.json`

```json
{
  "metadata": {
    "documents": ["sample1.pdf", "sample2.pdf"],
    "persona": "Startup Founder",
    "job": "Exploring AI for automating hiring",
    "timestamp": "2025-07-27T14:23:12.932Z"
  },
  "extracted_sections": [
    {
      "document": "sample1.pdf",
      "page": 3,
      "section_title": "AI in Recruitment Tools",
      "importance_rank": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "sample1.pdf",
      "page": 3,
      "refined_text": "This section explores how AI tools like resume parsing and candidate scoring...",
      "importance_rank": 1
    }
  ]
}
```

---

## 🚀 Running the Script

From the root of your project (where `processor.py` exists), run:

```bash
python3 processor.py
```

Ensure that:
- Your PDFs and `task.json` are placed in `input/`
- Output will be saved in `output/output.json`

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
PyMuPDF
numpy
scikit-learn
sentence-transformers
```

---

## ⚙️ Model Loading Behavior

The script attempts model loading in this order:

1. Look for models inside `/app/models` (including `.bin` or `.safetensors` files).
2. Fallback to common Hugging Face-compatible paths.
3. If not found, downloads model to `/app/models` cache directory.

If the model is **not available or fails to load**, Round 1B **will not run**.

---

## 🧪 Example Use Case

**Persona**: `"Product Manager at a fintech startup"`  
**Job**: `"Interested in implementing AI for customer onboarding"`

**PDFs**: Contain product brochures, case studies, whitepapers, etc.

**Output**: Top 10 sections from the PDFs relevant to AI-based onboarding, with semantic ranking.

---

## 🛑 Note

- Round 1B **requires an AI model**. If not loaded properly, it skips execution.
- Ensure PDFs have readable digital text (not scanned images) for best results.

---
