#!/usr/bin/env python3
"""
Embedding Model Comparison Script
=================================
Compares multiple embedding models using fair evaluation framework.

Key Features:
- Ensemble ground truth: Combines keyword matches + semantic matches from ALL models
- Fair comparison: All models contribute to ground truth, all evaluated equally
- Robust ISBN extraction: Handles various document formats
- Comprehensive metrics: Precision@5, Precision@10, Recall@5, Recall@10
- Statistical significance testing
"""

import os
import pandas as pd
import numpy as np
from typing import List, Set, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
import torch
from scipy import stats
import re

print("=" * 80)
print("EMBEDDING MODEL COMPARISON - FAIR EVALUATION")
print("=" * 80)

# ============================================================================
# Configuration: Models to Compare
# ============================================================================

EMBEDDING_MODELS = {
    "bge-small-en-v1.5": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "description": "Current model - Fast, 384 dims, good quality",
        "normalize": True
    },
    "all-MiniLM-L6-v2": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Very fast, 384 dims, smaller model",
        "normalize": True
    },
    "all-mpnet-base-v2": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "description": "Better quality, 768 dims, slower",
        "normalize": True
    },
}

# Test queries
QUERIES = {
    "mystery detective crime novels": ["mystery", "detective", "crime", "murder", "investigation"],
    "romantic love stories": ["romance", "love", "romantic", "relationship", "passion"],
    "fantasy magic adventure": ["fantasy", "magic", "magical", "wizard", "quest"],
    "science fiction space": ["science fiction", "sci-fi", "space", "future", "alien"],
    "historical fiction": ["historical", "history", "past", "ancient", "era"],
    "horror thriller suspense": ["horror", "thriller", "suspense", "scary", "terror"],
    "war military fiction": ["war", "military", "soldier", "battle", "army"],
    "family relationship": ["family", "father", "mother", "son", "daughter"],
}

# ============================================================================
# Load Book Dataset
# ============================================================================

print("\n📚 Loading book dataset...")
book_dataset = pd.read_csv("books_with_emotions.csv")
print(f"✓ Loaded {len(book_dataset)} books")

# Prepare TF-IDF baseline (same for all models)
books_tfidf = book_dataset[book_dataset['description'].notna()].copy()
books_tfidf['text'] = (books_tfidf['title'].fillna('') + ' ' + 
                       books_tfidf['description'].fillna('') + ' ' +
                       books_tfidf['categories'].fillna(''))

tfidf_vec = TfidfVectorizer(max_features=5000, stop_words='english', 
                            ngram_range=(1,2), min_df=2, max_df=0.8)
tfidf_mat = tfidf_vec.fit_transform(books_tfidf['text'])

# ============================================================================
# Create Vector Databases for Each Model
# ============================================================================

def create_vector_database(model_key: str, model_config: Dict, book_descriptions_file: str = "book_descriptions.txt"):
    """Create or load vector database for a specific embedding model"""
    db_path = f"./chroma_db_{model_key}"
    
    # Check if database already exists and has reasonable number of documents
    if os.path.exists(db_path) and os.listdir(db_path):
        print(f"  ✓ Loading existing database for {model_key}...")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_config["model_name"],
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": model_config["normalize"]}
        )
        vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        doc_count = vector_store._collection.count()
        print(f"  ✓ Loaded {doc_count} documents")
        
        # If database has very few documents, it's likely corrupted - recreate it
        if doc_count < 100:
            print(f"  ⚠️  Database has only {doc_count} documents, recreating...")
            import shutil
            import time
            try:
                # Close connections
                del vector_store
                time.sleep(0.5)
                shutil.rmtree(db_path)
                time.sleep(0.5)
            except Exception as e:
                print(f"    Warning: Could not delete {db_path}: {e}")
                print(f"    Please delete it manually and rerun the script")
                raise
        else:
            return vector_store, embeddings
    
    # Create new database
    print(f"  ⚙️  Creating new database for {model_key}...")
    print(f"     Model: {model_config['model_name']}")
    print(f"     This may take a few minutes...")
    
    # Load documents with robust splitting
    try:
        raw_docs = TextLoader(book_descriptions_file, encoding="utf-8").load()
        
        # Try CharacterTextSplitter first
        text_splitter = CharacterTextSplitter(
            chunk_size=0,
            chunk_overlap=0,
            separator="\n"
        )
        documents = text_splitter.split_documents(raw_docs)
        
        # If splitting didn't work (only 1 document), use manual splitting
        if len(documents) <= 1 and len(raw_docs) > 0:
            print(f"  ⚠️  Text splitter only created {len(documents)} document(s), using manual split...")
            with open(book_descriptions_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            documents = [Document(page_content=line.strip(), metadata={}) for line in lines if line.strip()]
            print(f"  ✓ Manually split into {len(documents)} documents")
        
        print(f"  ✓ Loaded {len(documents)} documents")
        
    except Exception as e:
        print(f"  ❌ Error loading documents: {e}")
        raise
    
    # Initialize embeddings
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_config["model_name"],
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": model_config["normalize"]}
    )
    
    # Create vector database
    print(f"  ⚙️  Generating embeddings (this may take a while)...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_path
    )
    print(f"  ✓ Database created for {model_key} ({vector_store._collection.count()} documents)")
    return vector_store, embeddings

# ============================================================================
# Robust ISBN Extraction
# ============================================================================

def extract_isbn_from_content(content: str) -> int:
    """Extract ISBN from document content with robust parsing"""
    if not content or not content.strip():
        return None
    
    # Remove leading quotes
    content = content.strip()
    if content.startswith('"'):
        content = content[1:].strip()
    if content.startswith("'"):
        content = content[1:].strip()
    
    # Try to get first token (should be ISBN)
    parts = content.split(None, 1)  # Split on first whitespace
    if parts:
        isbn_str = parts[0].strip('"').strip("'").strip()
        isbn_str = isbn_str.rstrip('"').rstrip("'")
        
        # Try direct conversion
        try:
            isbn_int = int(isbn_str)
            # Validate it's a reasonable ISBN (10-13 digits)
            if 10 <= len(str(isbn_int)) <= 13:
                return isbn_int
        except ValueError:
            pass
        
        # Try regex extraction
        isbn_match = re.search(r'\b\d{10,13}\b', content)
        if isbn_match:
            try:
                isbn_int = int(isbn_match.group())
                return isbn_int
            except ValueError:
                pass
    
    return None

# ============================================================================
# Search Functions
# ============================================================================

def semantic_search(query: str, vector_store, k: int = 10) -> List[int]:
    """
    Semantic search using vector store.
    Returns list of ISBNs in order of relevance.
    """
    try:
        # Get more results to account for potential extraction failures
        similar_results = vector_store.similarity_search_with_score(query, k=k*2)
        isbn_codes = []
        
        for doc, score in similar_results:
            isbn = extract_isbn_from_content(doc.page_content)
            if isbn is not None:
                isbn_codes.append(isbn)
                if len(isbn_codes) >= k:
                    break
        
        return isbn_codes[:k]
    except Exception as e:
        print(f"    Error in semantic search: {e}")
        import traceback
        traceback.print_exc()
        return []

def tfidf_search(query: str, k: int = 10) -> List[int]:
    """TF-IDF baseline search"""
    qvec = tfidf_vec.transform([query])
    sims = cosine_similarity(qvec, tfidf_mat).flatten()
    return books_tfidf.iloc[sims.argsort()[-k:][::-1]]['isbn13'].tolist()

# ============================================================================
# Generate Ensemble Ground Truth (FAIR for all models)
# ============================================================================

def generate_keyword_ground_truth(query: str, keywords: List[str], top_n: int = 60) -> Set[int]:
    """Generate keyword-based ground truth"""
    keyword_books = []
    for _, row in book_dataset.iterrows():
        text = f"{row.get('title','')} {row.get('description','')} {row.get('categories','')}".lower()
        score = sum(3 if kw in str(row.get('title','')).lower() else 
                   (1 if kw in text else 0) for kw in keywords)
        if score > 0:
            keyword_books.append((row['isbn13'], score))
    
    keyword_books.sort(key=lambda x: x[1], reverse=True)
    return set([isbn for isbn, _ in keyword_books[:top_n]])

def generate_ensemble_ground_truth(query: str, keywords: List[str], 
                                   all_vector_stores: Dict[str, Chroma],
                                   semantic_top_k: int = 5) -> Set[int]:
    """
    Generate ensemble ground truth that includes semantic matches from ALL models.
    This ensures fair comparison - all models contribute to ground truth.
    """
    # Part 1: Keyword-based matches (fair for all)
    keyword_isbns = generate_keyword_ground_truth(query, keywords, top_n=60)
    
    # Part 2: Semantic matches from ALL models (fair ensemble)
    semantic_isbns = set()
    for model_key, vector_store in all_vector_stores.items():
        try:
            # Get top results from each model
            sem_results = semantic_search(query, vector_store, k=semantic_top_k)
            semantic_isbns.update(sem_results)
        except Exception as e:
            print(f"    Warning: Could not get semantic results from {model_key}: {e}")
            continue
    
    # Combine: This creates a comprehensive ground truth that includes
    # both obvious keyword matches and semantic matches from all models
    return keyword_isbns | semantic_isbns

# ============================================================================
# Metrics
# ============================================================================

def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Calculate Precision@K"""
    if k == 0:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / k

def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Calculate Recall@K"""
    if len(relevant) == 0:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / len(relevant)

# ============================================================================
# Evaluation for Each Model
# ============================================================================

def evaluate_model(model_key: str, model_config: Dict, ground_truth: Dict[str, Set[int]]) -> pd.DataFrame:
    """Evaluate a single embedding model"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_key}")
    print(f"Model: {model_config['model_name']}")
    print(f"Description: {model_config['description']}")
    print(f"{'='*80}")
    
    # Create/load vector database
    vector_store, embeddings = create_vector_database(model_key, model_config)
    
    # Run evaluation using shared ground truth
    print(f"\n🔍 Running evaluation (using ensemble ground truth)...")
    results = []
    
    for query, gt in ground_truth.items():
        sem = semantic_search(query, vector_store, 10)
        tfidf = tfidf_search(query, 10)
        
        # Calculate all metrics
        sem_p5 = precision_at_k(sem, gt, 5)
        tfidf_p5 = precision_at_k(tfidf, gt, 5)
        sem_p10 = precision_at_k(sem, gt, 10)
        tfidf_p10 = precision_at_k(tfidf, gt, 10)
        
        sem_r5 = recall_at_k(sem, gt, 5)
        tfidf_r5 = recall_at_k(tfidf, gt, 5)
        sem_r10 = recall_at_k(sem, gt, 10)
        tfidf_r10 = recall_at_k(tfidf, gt, 10)
        
        results.append({
            'model': model_key,
            'query': query,
            'gt_size': len(gt),
            'semantic_P@5': sem_p5,
            'tfidf_P@5': tfidf_p5,
            'semantic_P@10': sem_p10,
            'tfidf_P@10': tfidf_p10,
            'semantic_R@5': sem_r5,
            'tfidf_R@5': tfidf_r5,
            'semantic_R@10': sem_r10,
            'tfidf_R@10': tfidf_r10,
        })
        
        sem_found = len(set(sem) & gt)
        print(f"  {query[:30]:<30} | P@10: S={sem_p10:>4.0%} T={tfidf_p10:>4.0%} | R@10: S={sem_r10:>4.0%} T={tfidf_r10:>4.0%} | Found: {sem_found}/{len(gt)}")
    
    df = pd.DataFrame(results)
    
    # Calculate averages
    sem_p10_avg = df['semantic_P@10'].mean()
    tfidf_p10_avg = df['tfidf_P@10'].mean()
    sem_r10_avg = df['semantic_R@10'].mean()
    tfidf_r10_avg = df['tfidf_R@10'].mean()
    
    improvement_p = ((sem_p10_avg - tfidf_p10_avg) / tfidf_p10_avg * 100) if tfidf_p10_avg > 0 else 0
    improvement_r = ((sem_r10_avg - tfidf_r10_avg) / tfidf_r10_avg * 100) if tfidf_r10_avg > 0 else 0
    
    print(f"\n📈 Results for {model_key}:")
    print(f"   Precision@10: {sem_p10_avg:.1%} (vs TF-IDF: {tfidf_p10_avg:.1%}, improvement: {improvement_p:+.1f}%)")
    print(f"   Recall@10:    {sem_r10_avg:.1%} (vs TF-IDF: {tfidf_r10_avg:.1%}, improvement: {improvement_r:+.1f}%)")
    
    return df

# ============================================================================
# Main Comparison
# ============================================================================

print("\n" + "="*80)
print("STARTING EMBEDDING MODEL COMPARISON")
print("="*80)
print(f"\nModels to compare: {len(EMBEDDING_MODELS)}")
for key, config in EMBEDDING_MODELS.items():
    print(f"  - {key}: {config['description']}")

print(f"\n⚠️  NOTE: Using ENSEMBLE ground truth (keyword + semantic from ALL models)")
print(f"⚠️  NOTE: Query understanding is DISABLED to test embedding models only")
print(f"⚠️  NOTE: All models should outperform TF-IDF baseline")

# Step 1: Create all vector databases first
print("\n" + "="*80)
print("STEP 1: Creating/loading vector databases for all models")
print("="*80)
all_vector_stores = {}
for model_key, model_config in EMBEDDING_MODELS.items():
    try:
        vector_store, _ = create_vector_database(model_key, model_config)
        all_vector_stores[model_key] = vector_store
    except Exception as e:
        print(f"\n❌ Error creating database for {model_key}: {e}")
        import traceback
        traceback.print_exc()
        print("   Skipping this model...")
        continue

if not all_vector_stores:
    print("\n❌ No vector databases were successfully created!")
    exit(1)

# Step 2: Generate ensemble ground truth
print("\n" + "="*80)
print("STEP 2: Generating ensemble ground truth (fair for all models)")
print("="*80)
ground_truth = {}
for query, keywords in QUERIES.items():
    gt = generate_ensemble_ground_truth(query, keywords, all_vector_stores, semantic_top_k=5)
    ground_truth[query] = gt
    print(f"✓ {query[:40]:<40} | {len(gt)} books (keyword + semantic from all models)")

# Step 3: Evaluate each model
print("\n" + "="*80)
print("STEP 3: Evaluating all models")
print("="*80)

all_results = []
for model_key, model_config in EMBEDDING_MODELS.items():
    if model_key not in all_vector_stores:
        print(f"\n⚠️  Skipping {model_key} (database not available)")
        continue
    try:
        df = evaluate_model(model_key, model_config, ground_truth)
        all_results.append(df)
    except Exception as e:
        print(f"\n❌ Error evaluating {model_key}: {e}")
        import traceback
        traceback.print_exc()
        print("   Skipping this model...")
        continue

# Step 4: Analyze results
if all_results:
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save detailed results
    combined_df.to_csv('embedding_model_comparison_results.csv', index=False)
    print(f"\n✓ Detailed results saved: embedding_model_comparison_results.csv")
    
    # Calculate summary statistics
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    summary = []
    for model_key in EMBEDDING_MODELS.keys():
        model_df = combined_df[combined_df['model'] == model_key]
        if len(model_df) > 0:
            sem_p10 = model_df['semantic_P@10'].mean()
            tfidf_p10 = model_df['tfidf_P@10'].mean()
            sem_r10 = model_df['semantic_R@10'].mean()
            tfidf_r10 = model_df['tfidf_R@10'].mean()
            
            improvement_p = ((sem_p10 - tfidf_p10) / tfidf_p10 * 100) if tfidf_p10 > 0 else 0
            improvement_r = ((sem_r10 - tfidf_r10) / tfidf_r10 * 100) if tfidf_r10 > 0 else 0
            
            # Statistical significance vs TF-IDF
            t_stat_p, p_val_p = stats.ttest_rel(model_df['semantic_P@10'], model_df['tfidf_P@10'])
            t_stat_r, p_val_r = stats.ttest_rel(model_df['semantic_R@10'], model_df['tfidf_R@10'])
            
            summary.append({
                'Model': model_key,
                'Model Name': EMBEDDING_MODELS[model_key]['model_name'],
                'P@10': f"{sem_p10:.1%}",
                'vs TF-IDF P@10': f"{tfidf_p10:.1%}",
                'P@10 Improvement': f"{improvement_p:+.1f}%",
                'R@10': f"{sem_r10:.1%}",
                'vs TF-IDF R@10': f"{tfidf_r10:.1%}",
                'R@10 Improvement': f"{improvement_r:+.1f}%",
                'P@10 p-value': f"{p_val_p:.4f}",
                'P@10 Significant': '✓' if p_val_p < 0.05 else '✗',
                'R@10 p-value': f"{p_val_r:.4f}",
                'R@10 Significant': '✓' if p_val_r < 0.05 else '✗'
            })
    
    summary_df = pd.DataFrame(summary)
    print("\n" + summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv('embedding_model_comparison_summary.csv', index=False)
    print(f"\n✓ Summary saved: embedding_model_comparison_summary.csv")
    
    # Find best model
    if len(summary_df) > 0:
        # Extract numeric values
        summary_df['P@10_num'] = summary_df['P@10'].str.rstrip('%').astype(float)
        summary_df['R@10_num'] = summary_df['R@10'].str.rstrip('%').astype(float)
        
        # Best by precision
        best_p_idx = summary_df['P@10_num'].idxmax()
        best_p_model = summary_df.loc[best_p_idx]
        
        # Best by recall
        best_r_idx = summary_df['R@10_num'].idxmax()
        best_r_model = summary_df.loc[best_r_idx]
        
        print(f"\n🏆 Best Model by Precision@10: {best_p_model['Model']} ({best_p_model['Model Name']})")
        print(f"   Precision@10: {best_p_model['P@10']}")
        print(f"   Improvement: {best_p_model['P@10 Improvement']}")
        
        print(f"\n🏆 Best Model by Recall@10: {best_r_model['Model']} ({best_r_model['Model Name']})")
        print(f"   Recall@10: {best_r_model['R@10']}")
        print(f"   Improvement: {best_r_model['R@10 Improvement']}")
        
        # Check if all models beat TF-IDF (both precision and recall)
        tfidf_p10 = summary_df['vs TF-IDF P@10'].str.rstrip('%').astype(float)
        tfidf_r10 = summary_df['vs TF-IDF R@10'].str.rstrip('%').astype(float)
        
        all_better_p = (summary_df['P@10_num'] > tfidf_p10).all()
        all_better_r = (summary_df['R@10_num'] > tfidf_r10).all()
        all_better = all_better_p and all_better_r
        
        print(f"\n{'✅' if all_better else '⚠️ '} All embedding models {'outperform' if all_better else 'do not all outperform'} TF-IDF baseline")
        if all_better_p:
            print(f"   ✓ All models have higher Precision@10 than TF-IDF")
        if all_better_r:
            print(f"   ✓ All models have higher Recall@10 than TF-IDF")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
else:
    print("\n❌ No models were successfully evaluated!")
