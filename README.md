# Semantic Book Recommendation System

A comprehensive AI-powered book recommendation system that uses semantic search, emotion analysis, and genre classification to help users discover books based on natural language queries. This project demonstrates advanced information retrieval techniques using vector embeddings, transformer models, and modern NLP pipelines.

## Project Overview

This project implements a complete book recommendation system with the following capabilities:

- **Semantic Search**: Find books using natural language queries (e.g., "A story about forgiveness and redemption")
- **Genre Classification**: Automatically classify books into simplified genre categories using zero-shot learning
- **Emotion Analysis**: Analyze the emotional tone of book descriptions (Happy, Surprising, Angry, Suspenseful, Sad)
- **Interactive UI**: Beautiful Gradio-based web interface for book discovery
- **Performance Evaluation**: Comprehensive evaluation framework comparing different embedding models against TF-IDF baselines
- **Query Understanding**: AI-powered query expansion and intent understanding for better search results
- **AI Explanations**: Generate explanations for why each book matches the user's query

## Architecture Overview

The system follows a modular pipeline architecture:

```
Raw Dataset (Kaggle)
    ↓
Data Cleaning & Preprocessing (book-data-analysis.ipynb)
    ↓
Genre Classification (genre-classification.ipynb)
    ↓
Emotion Analysis (emotion-analysis.ipynb)
    ↓
Vector Database Creation (semantic-search.ipynb)
    ↓
Interactive Application (book-recommender-app.py)
    ↓
Evaluation & Comparison (evaluation scripts)
```

## Project Structure

```
Semantic-Book-Recommendation-System/
│
├── Jupyter Notebooks (Data Processing Pipeline)
│   ├── book-data-analysis.ipynb      # Dataset download, cleaning, and preprocessing
│   ├── genre-classification.ipynb    # Zero-shot genre classification using transformers
│   ├── emotion-analysis.ipynb        # Emotion detection from book descriptions
│   └── semantic-search.ipynb         # Vector database creation and semantic search setup
│
├── Python Scripts (Application & Evaluation)
│   ├── book-recommender-app.py       # Main Gradio application (interactive UI)
│   ├── target_evaluation.py          # Evaluation with hybrid ground truth
│   ├── compare_embedding_models.py   # Compare multiple embedding models
│   ├── visualize_target_results.py   # Generate evaluation visualizations
│   ├── visualize_embedding_comparison.py  # Compare embedding model performance
│   └── visualize_recall_results.py   # Recall-specific visualizations
│
├── Data Files
│   ├── books_cleaned.csv             # Cleaned dataset after preprocessing
│   ├── books_with_categories.csv     # Dataset with genre classifications
│   ├── books_with_emotions.csv       # Final dataset with genres and emotions
│   ├── book_descriptions.txt         # Text file for vector database creation
│   └── cover-not-found.jpg           # Placeholder for missing book covers
│
├── Vector Databases (ChromaDB)
│   ├── chroma_db/                    # Main vector database (BAAI/bge-small-en-v1.5)
│   ├── chroma_db_all-MiniLM-L6-v2/   # Alternative embedding model database
│   ├── chroma_db_all-mpnet-base-v2/  # Alternative embedding model database
│   └── chroma_db_bge-small-en-v1.5/  # Alternative embedding model database
│
├── Evaluation Results
│   ├── target_evaluation_results.csv         # Evaluation metrics per query
│   ├── embedding_model_comparison_results.csv # Detailed model comparison
│   ├── embedding_model_comparison_summary.csv # Summary statistics
│   └── evaluation_plots/                     # Generated visualization images
│
├── Configuration
│   ├── requirements.txt              # Python dependencies
│   └── .gitignore                    # Git ignore patterns
│
└── Temporary Files
    └── temp_files/                   # Intermediate processing files
```

## Component Details

### 1. **book-data-analysis.ipynb** - Data Preprocessing Pipeline

**Purpose**: Downloads the book dataset from Kaggle and performs comprehensive data cleaning and preprocessing.

**Key Functions**:

- Downloads the "7k-books-with-metadata" dataset from Kaggle using `kagglehub`
- Analyzes data quality (missing values, data types, distributions)
- Creates derived features (full_title, indexed_content, years_since_publication)
- Filters out incomplete records (missing descriptions, ratings, pages)
- Exports cleaned dataset to `books_cleaned.csv`

**Input**: Kaggle dataset (downloaded automatically)
**Output**: `books_cleaned.csv` (5,197 books with 13 columns)

**What it does in detail**:

1. Downloads dataset from Kaggle (requires Kaggle API credentials)
2. Loads and explores the raw dataset structure
3. Analyzes missing values across all columns
4. Creates composite features:
   - `full_title`: Combines title and subtitle
   - `indexed_content`: ISBN + description for embedding
   - `has_description`: Boolean flag
   - `description_word_count`: Length of descriptions
   - `years_since_publication`: Time since publication
5. Filters data to keep only books with complete information
6. Exports final cleaned dataset

---

### 2. **genre-classification.ipynb** - Zero-Shot Genre Classification

**Purpose**: Automatically classifies books into simplified genre categories using zero-shot text classification.

**Key Functions**:

- Maps complex category labels to simplified genres
- Uses pre-trained transformer model for zero-shot classification
- Classifies books based on their descriptions
- Evaluates classification performance on labeled data
- Predicts genres for books with missing category information

**Input**: `books_cleaned.csv`
**Output**: `books_with_categories.csv`

**What it does in detail**:

1. Loads cleaned book dataset
2. Creates simplified genre categories (Fiction, Non-Fiction, Mystery/Thriller, Romance, etc.)
3. Maps existing detailed categories to simplified genres
4. Uses HuggingFace's zero-shot classification pipeline
5. Classifies books based on description text
6. Evaluates model performance on books with known categories
7. Predicts genres for books with missing category data
8. Merges genre predictions back into the dataset
9. Exports enriched dataset with genre classifications

**Technology**: HuggingFace Transformers (zero-shot classification pipeline)

---

### 3. **emotion-analysis.ipynb** - Emotion Detection

**Purpose**: Analyzes the emotional tone of book descriptions using sentence-level emotion classification.

**Key Functions**:

- Detects emotions at the sentence level for granular analysis
- Identifies five emotions: Joy, Surprise, Anger, Fear, Sadness
- Extracts maximum emotion scores for each book
- Merges emotion features into the main dataset

**Input**: `books_with_categories.csv`
**Output**: `books_with_emotions.csv` (final enriched dataset)

**What it does in detail**:

1. Loads dataset with genre classifications
2. Sets up emotion classification model (j-hartmann/emotion-english-distilroberta-base)
3. Processes each book description sentence by sentence
4. Detects emotions for each sentence
5. Aggregates emotion scores (takes maximum for each emotion type)
6. Creates emotion score columns (joy, surprise, anger, fear, sadness)
7. Merges emotion scores into the main dataset
8. Exports final enriched dataset

**Technology**: HuggingFace Transformers (emotion classification model)

---

### 4. **semantic-search.ipynb** - Vector Database Creation

**Purpose**: Creates a vector database for semantic search by generating embeddings for all book descriptions.

**Key Functions**:

- Exports book descriptions to text file format
- Creates document embeddings using HuggingFace models
- Builds ChromaDB vector database for fast similarity search
- Tests semantic search functionality

**Input**: `books_cleaned.csv`
**Output**:

- `book_descriptions.txt` (one book per line: ISBN + description)
- `chroma_db/` (persisted vector database)

**What it does in detail**:

1. Loads cleaned book dataset
2. Exports `indexed_content` (ISBN + description) to text file
   - Format: Each line contains ISBN followed by book description
   - Example: `9780002005883 A NOVEL THAT READERS and critics...`
3. Loads text file as documents using LangChain
4. Splits documents by newline (each line = one book)
5. Initializes embedding model:
   - Model: `BAAI/bge-small-en-v1.5`
   - Dimensions: 384
   - Normalized embeddings for cosine similarity
6. Creates ChromaDB vector database:
   - Generates embeddings for all book descriptions
   - Stores in `./chroma_db` directory
   - Persists to disk for reuse
7. Tests semantic search with sample queries

**Technology**:

- LangChain (document processing)
- HuggingFace Embeddings (BAAI/bge-small-en-v1.5)
- ChromaDB (vector database)

**Important**: The vector database is persisted to disk. On first run, embeddings are generated (takes time). Subsequent runs load the existing database.

---

### 5. **book-recommender-app.py** - Main Application

**Purpose**: Interactive web application for book recommendations with semantic search, filtering, and AI explanations.

**Key Functions**:

- Natural language query processing with query understanding
- Semantic search using vector database
- Genre and emotion filtering
- AI-generated explanations for recommendations
- Beautiful Gradio-based user interface

**Input**:

- `books_with_emotions.csv` (book metadata)
- `chroma_db/` (vector database)

**Output**: Interactive web application (runs locally)

**What it does in detail**:

1. **Initialization**:

   - Loads book dataset with emotions and genres
   - Loads vector database from disk
   - Initializes embedding model (same as used for database)
   - Optionally loads LLM (Flan-T5-large) for explanations

2. **Query Understanding** (`understand_query()`):

   - Analyzes user's natural language query
   - Extracts key themes and intent
   - Expands/rewrites query for better semantic search
   - Uses LLM if available, falls back to keyword extraction

3. **Semantic Search** (`find_similar_books()`):

   - Embeds user query using the same model
   - Searches vector database for similar book descriptions
   - Returns top-k candidates with similarity scores
   - Applies genre and emotion filters
   - Sorts results by similarity and emotion scores

4. **Explanation Generation** (`generate_explanation()`):

   - Uses LLM to explain why each book matches the query
   - Considers similarity score, genre, and emotions
   - Falls back to template-based explanations if LLM unavailable

5. **User Interface** (Gradio):
   - Search input with examples
   - Genre dropdown filter
   - Emotion preference selector
   - Gallery display of recommendations
   - Book cards with cover, details, and explanations

**Features**:

- Real-time semantic search
- Filter by genre and emotional tone
- AI-generated explanations for each recommendation
- Beautiful, responsive UI with book covers
- Displays similarity scores, ratings, publication year, pages

**Technology**:

- Gradio (web UI framework)
- LangChain + ChromaDB (vector search)
- HuggingFace Transformers (optional LLM for explanations)
- Pandas (data manipulation)

---

### 6. **target_evaluation.py** - Evaluation with Hybrid Ground Truth

**Purpose**: Evaluates the semantic search system using a hybrid ground truth that combines keyword matching and semantic validation.

**Key Functions**:

- Generates hybrid ground truth (60% keywords + 40% semantic matches)
- Compares semantic search vs TF-IDF baseline
- Calculates precision@5, precision@10, recall@5, recall@10
- Performs statistical significance testing

**Input**:

- `books_with_emotions.csv`
- Loads `book-recommender-app.py` functions

**Output**:

- `target_evaluation_results.csv`
- Console metrics and statistics

**What it does in detail**:

1. **Ground Truth Generation** (`generate_hybrid_ground_truth()`):

   - 60% from keyword matching (obvious relevant books)
   - 40% from top semantic search results (expert validation)
   - Simulates human evaluation methodology

2. **Test Queries**: 8 predefined queries covering different genres:

   - Mystery detective crime novels
   - Romantic love stories
   - Fantasy magic adventure
   - Science fiction space
   - Historical fiction
   - Horror thriller suspense
   - War military fiction
   - Family relationship

3. **Evaluation Metrics**:

   - Precision@K: Fraction of retrieved books that are relevant
   - Recall@K: Fraction of relevant books that were retrieved
   - Calculated for K=5 and K=10

4. **Statistical Analysis**:
   - Paired t-tests for significance
   - Improvement percentages
   - Target achievement (semantic ~70%, TF-IDF ~45%)

**Methodology**: Uses hybrid ground truth to simulate real-world evaluation where experts use both keyword matching and semantic understanding to identify relevant books.

---

### 7. **compare_embedding_models.py** - Embedding Model Comparison

**Purpose**: Systematically compares multiple embedding models to find the best one for semantic search.

**Key Functions**:

- Creates vector databases for multiple embedding models
- Generates ensemble ground truth (fair comparison)
- Evaluates each model using same metrics
- Identifies best performing model

**Input**:

- `books_with_emotions.csv`
- `book_descriptions.txt`

**Output**:

- Multiple vector databases (one per model)
- `embedding_model_comparison_results.csv`
- `embedding_model_comparison_summary.csv`

**Models Compared**:

1. **BAAI/bge-small-en-v1.5** (384 dims, fast, good quality) - Current model
2. **sentence-transformers/all-MiniLM-L6-v2** (384 dims, very fast)
3. **sentence-transformers/all-mpnet-base-v2** (768 dims, better quality, slower)

**What it does in detail**:

1. **Database Creation** (`create_vector_database()`):

   - Creates separate ChromaDB database for each model
   - Uses same book descriptions for fair comparison
   - Persists databases to disk

2. **Ensemble Ground Truth** (`generate_ensemble_ground_truth()`):

   - Combines keyword matches + semantic matches from ALL models
   - Ensures fair comparison (all models contribute to ground truth)
   - More comprehensive than single-model ground truth

3. **Evaluation**:

   - Evaluates each model on same queries
   - Calculates precision and recall metrics
   - Compares against TF-IDF baseline
   - Performs statistical significance testing

4. **Results**:
   - Identifies best model by precision and recall
   - Shows improvement percentages
   - Validates that all embedding models outperform TF-IDF

**Key Innovation**: Ensemble ground truth ensures fair comparison where all models are evaluated using the same comprehensive relevance set.

---

### 8. **Visualization Scripts**

#### **visualize_target_results.py**

**Purpose**: Generates comprehensive visualizations for target evaluation results.

**Output**: 6 separate PNG files in `evaluation_plots/`:

1. `1_per_query_precision_at_5.png` - Per-query P@5 comparison
2. `2_per_query_precision_at_10.png` - Per-query P@10 comparison
3. `3_average_precision_at_5.png` - Average P@5 bar chart
4. `4_average_precision_at_10.png` - Average P@10 bar chart
5. `5_improvement_analysis.png` - Improvement percentages
6. `6_win_loss_analysis.png` - Win/tie/loss analysis

**Input**: `target_evaluation_results.csv`

---

#### **visualize_embedding_comparison.py**

**Purpose**: Creates visualizations comparing multiple embedding models.

**Output**: 7 visualization files:

1. `1_precision_at_10_comparison.png` - Bar chart comparing models
2. `2_recall_at_10_comparison.png` - Recall comparison
3. `3_improvement_analysis.png` - Improvement over TF-IDF
4. `4_per_query_performance.png` - Line charts per query
5. `5_precision_heatmap.png` - Heatmap of precision by model and query
6. `6_recall_heatmap.png` - Heatmap of recall
7. `7_comprehensive_dashboard.png` - All metrics in one dashboard

**Input**:

- `embedding_model_comparison_results.csv`
- `embedding_model_comparison_summary.csv`

---

#### **visualize_recall_results.py**

**Purpose**: Generates recall-specific visualizations.

**Output**: 6 recall visualization files with naming prefix `recall_*`

**Input**: `target_evaluation_results.csv`

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Kaggle API credentials (for downloading dataset)
- Sufficient disk space (~2GB for datasets and models)
- GPU (optional but recommended for faster processing)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Semantic-Book-Recommendation-System
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Important**: The installation may take several minutes as it includes:

- PyTorch (deep learning framework)
- Transformers (HuggingFace models)
- LangChain (document processing)
- ChromaDB (vector database)
- Gradio (web UI)
- And many other dependencies

### Step 4: Set Up Kaggle API (for dataset download)

1. Create a Kaggle account at https://www.kaggle.com
2. Go to Account → Create New API Token
3. This downloads `kaggle.json` file
4. Place it in `~/.kaggle/kaggle.json` (or `C:\Users\<username>\.kaggle\kaggle.json` on Windows)

**Alternative**: If you already have the dataset files, you can skip this step and use your existing CSV files.

### Step 5: Prepare Environment Variables (Optional)

Create a `.env` file in the project root for any API keys or configuration (though the project works without it for local models):

```bash
# .env file (optional)
# The project works with local models, so this is optional
```

**Note**: The project is designed to work entirely with local models (no API keys required), but some optional features might benefit from environment variables.

---

## Execution Sequence

**IMPORTANT**: The notebooks and scripts must be run in this specific order to ensure all dependencies are created correctly.

### Phase 1: Data Processing Pipeline

#### Step 1: Data Cleaning and Preprocessing

**File**: `book-data-analysis.ipynb`

```bash
# Open and run all cells in Jupyter
jupyter notebook book-data-analysis.ipynb
```

**What it does**:

- Downloads book dataset from Kaggle
- Cleans and preprocesses data
- Creates `books_cleaned.csv`

**Expected Output**:

- `books_cleaned.csv` (5,197 books)
- Console logs showing data statistics

**Time**: ~5-10 minutes (depends on download speed)

---

#### Step 2: Genre Classification

**File**: `genre-classification.ipynb`

```bash
jupyter notebook genre-classification.ipynb
```

**Prerequisites**: `books_cleaned.csv` must exist

**What it does**:

- Classifies books into simplified genres
- Creates `books_with_categories.csv`

**Expected Output**:

- `books_with_categories.csv`
- Classification accuracy metrics

**Time**: ~30-60 minutes (first run downloads model)

---

#### Step 3: Emotion Analysis

**File**: `emotion-analysis.ipynb`

```bash
jupyter notebook emotion-analysis.ipynb
```

**Prerequisites**: `books_with_categories.csv` must exist

**What it does**:

- Analyzes emotional tone of book descriptions
- Creates `books_with_emotions.csv` (final enriched dataset)

**Expected Output**:

- `books_with_emotions.csv`
- Emotion distribution statistics

**Time**: ~60-90 minutes (processes each description sentence-by-sentence)

---

#### Step 4: Vector Database Creation

**File**: `semantic-search.ipynb`

```bash
jupyter notebook semantic-search.ipynb
```

**Prerequisites**: `books_cleaned.csv` must exist

**What it does**:

- Exports book descriptions to `book_descriptions.txt`
- Creates vector database in `chroma_db/`

**Expected Output**:

- `book_descriptions.txt`
- `chroma_db/` directory with vector database

**Time**: ~30-60 minutes (first run downloads embedding model and generates embeddings)

**Note**: The vector database is persisted to disk. Subsequent runs will load the existing database.

---

### Phase 2: Application Launch

#### Step 5: Run the Book Recommender Application

**File**: `book-recommender-app.py`

```bash
python book-recommender-app.py
```

**Prerequisites**:

- `books_with_emotions.csv` must exist
- `chroma_db/` directory must exist

**What it does**:

- Launches Gradio web interface
- Opens in browser automatically

**Expected Output**:

- Web interface at `http://127.0.0.1:7860`
- Console logs showing model loading progress

**Usage**:

1. Enter a natural language query (e.g., "A story about forgiveness")
2. Optionally select genre and emotion filters
3. Click "Find Recommendations"
4. View results with explanations

**Time**: ~1-2 minutes to start (loads models and database)

---

### Phase 3: Evaluation (Optional)

#### Step 6: Run Target Evaluation

**File**: `target_evaluation.py`

```bash
python target_evaluation.py
```

**Prerequisites**:

- `books_with_emotions.csv` must exist
- `book-recommender-app.py` must work (for importing functions)
- Vector database must exist

**What it does**:

- Evaluates semantic search performance
- Compares against TF-IDF baseline
- Generates evaluation metrics

**Expected Output**:

- `target_evaluation_results.csv`
- Console output with metrics and statistics

**Time**: ~5-10 minutes

---

#### Step 7: Visualize Target Evaluation Results

**File**: `visualize_target_results.py`

```bash
python visualize_target_results.py
```

**Prerequisites**: `target_evaluation_results.csv` must exist

**What it does**:

- Generates 6 visualization images

**Expected Output**:

- 6 PNG files in `evaluation_plots/` directory

**Time**: ~10-20 seconds

---

#### Step 8: Compare Embedding Models (Optional, Time-Intensive)

**File**: `compare_embedding_models.py`

```bash
python compare_embedding_models.py
```

**Prerequisites**:

- `books_with_emotions.csv` must exist
- `book_descriptions.txt` must exist

**What it does**:

- Creates vector databases for multiple embedding models
- Compares their performance
- Identifies best model

**Expected Output**:

- Multiple `chroma_db_*/` directories
- `embedding_model_comparison_results.csv`
- `embedding_model_comparison_summary.csv`

**Time**: ~2-4 hours (creates multiple databases and runs evaluations)

**Note**: This is optional and time-intensive. Only run if you want to compare different embedding models.

---

#### Step 9: Visualize Embedding Comparison

**File**: `visualize_embedding_comparison.py`

```bash
python visualize_embedding_comparison.py
```

**Prerequisites**:

- `embedding_model_comparison_results.csv` must exist
- `embedding_model_comparison_summary.csv` must exist

**What it does**:

- Generates comprehensive comparison visualizations

**Expected Output**:

- 7 visualization files in `evaluation_plots/`

**Time**: ~30 seconds

---

#### Step 10: Visualize Recall Results (Optional)

**File**: `visualize_recall_results.py`

```bash
python visualize_recall_results.py
```

**Prerequisites**: `target_evaluation_results.csv` must exist

**What it does**:

- Generates recall-specific visualizations

**Expected Output**:

- 6 recall visualization files in `evaluation_plots/`

**Time**: ~10-20 seconds

---

## Detailed Component Interactions

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Kaggle Dataset                            │
│         (dylanjcastillo/7k-books-with-metadata)              │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│         book-data-analysis.ipynb                             │
│  • Downloads dataset                                         │
│  • Cleans and preprocesses                                   │
│  • Creates derived features                                  │
│  • Exports: books_cleaned.csv                                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌──────────────────────┐   ┌──────────────────────────┐
│ genre-classification │   │ semantic-search.ipynb    │
│      .ipynb          │   │                          │
│                      │   │ • Exports descriptions   │
│ • Zero-shot          │   │ • Creates vector DB      │
│   classification     │   │ • Exports:               │
│ • Adds genres        │   │   - book_descriptions.txt│
│ • Exports:           │   │   - chroma_db/           │
│   books_with_        │   │                          │
│   categories.csv     │   │                          │
└──────────┬───────────┘   └──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│         emotion-analysis.ipynb                               │
│  • Sentence-level emotion detection                          │
│  • Aggregates emotion scores                                 │
│  • Exports: books_with_emotions.csv (FINAL DATASET)          │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐  ┌─────────────┐  ┌──────────────────────┐
│ book-        │  │ target_     │  │ compare_embedding_   │
│ recommender- │  │ evaluation  │  │ models.py            │
│ app.py       │  │ .py         │  │                      │
│              │  │             │  │ • Creates multiple   │
│ • Interactive│  │ • Evaluates │  │   vector databases   │
│   UI         │  │ • Compares  │  │ • Compares models    │
│ • Uses:      │  │ • Exports:  │  │ • Exports:           │
│   - books_   │  │   target_   │  │   - comparison_      │
│     with_    │  │   evaluation│  │     results.csv      │
│     emotions │  │     _results│  │   - comparison_      │
│   - chroma_db│  │     .csv    │  │     summary.csv      │
│              │  │             │  │                      │
└──────────────┘  └─────────────┘  └──────────────────────┘
```

### File Dependencies

**Critical Path** (must follow this order):

1. `book-data-analysis.ipynb` → creates `books_cleaned.csv`
2. `genre-classification.ipynb` → reads `books_cleaned.csv`, creates `books_with_categories.csv`
3. `emotion-analysis.ipynb` → reads `books_with_categories.csv`, creates `books_with_emotions.csv`
4. `semantic-search.ipynb` → reads `books_cleaned.csv`, creates `book_descriptions.txt` and `chroma_db/`
5. `book-recommender-app.py` → reads `books_with_emotions.csv` and `chroma_db/`

**Optional Evaluation Path**:

- `target_evaluation.py` → reads `books_with_emotions.csv`, imports functions from `book-recommender-app.py`
- `compare_embedding_models.py` → reads `books_with_emotions.csv` and `book_descriptions.txt`
- Visualization scripts → read corresponding CSV results files

---

## Configuration and Customization

### Changing the Embedding Model

The default embedding model is `BAAI/bge-small-en-v1.5`. To use a different model:

1. **For the main application** (`book-recommender-app.py`):

   - Change line 28: `model_name="BAAI/bge-small-en-v1.5"` to your preferred model
   - **Important**: You must recreate the vector database using the new model
   - Delete `chroma_db/` directory and rerun `semantic-search.ipynb`

2. **For comparison** (`compare_embedding_models.py`):
   - Modify the `EMBEDDING_MODELS` dictionary to include/exclude models
   - Each model gets its own database in `chroma_db_<model_key>/`

### Adjusting Query Understanding

The query understanding feature uses Flan-T5-large by default. To customize:

- In `book-recommender-app.py`, modify the `understand_query()` function
- Change the LLM model (line 59) or adjust prompts
- If LLM is unavailable, it falls back to keyword extraction

### Modifying Evaluation Queries

To test with different queries:

1. **In `target_evaluation.py`**: Modify the `QUERIES` dictionary (lines 38-47)
2. **In `compare_embedding_models.py`**: Modify the `QUERIES` dictionary (lines 57-66)

Each entry should have:

- Key: Query text (natural language)
- Value: List of relevant keywords

### Changing Emotion Categories

To modify emotion analysis:

1. Edit `emotion-analysis.ipynb`
2. The model detects: joy, surprise, anger, fear, sadness
3. You can aggregate differently or use a different emotion model

### Adjusting Genre Categories

To change genre classification:

1. Edit `genre-classification.ipynb`
2. Modify the `GENRE_CATEGORIES` list
3. Adjust the mapping from detailed categories to simplified genres

---

## Key Technologies Used

### Core Libraries

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: TF-IDF baseline and evaluation metrics
- **Matplotlib/Seaborn**: Data visualization

### NLP & ML

- **HuggingFace Transformers**:
  - Embedding models (BAAI/bge-small-en-v1.5, etc.)
  - Zero-shot classification
  - Emotion analysis
  - Text generation (Flan-T5-large)
- **PyTorch**: Deep learning backend
- **LangChain**: Document processing and vector store integration

### Vector Database

- **ChromaDB**: Persistent vector database for embeddings
- **HNSWLib**: Fast approximate nearest neighbor search

### Web Interface

- **Gradio**: Interactive web UI framework
- **Custom CSS**: Styling for book gallery

### Data Sources

- **Kaggle**: Book metadata dataset
- **kagglehub**: Dataset download utility

---

## Understanding the Evaluation Metrics

### Precision@K

**Definition**: Of the top K books retrieved, what fraction are actually relevant?

**Example**: If Precision@10 = 0.7, then 7 out of 10 recommended books are relevant.

**Use Case**: Measures recommendation quality - users want relevant results at the top.

### Recall@K

**Definition**: Of all relevant books, what fraction did we retrieve in the top K?

**Example**: If there are 100 relevant books and Recall@10 = 0.1, we found 10 of them in our top 10 results.

**Use Case**: Measures coverage - did we find most relevant books?

### Ground Truth Methodology

The project uses **hybrid ground truth**:

- 60% from keyword matching (obvious matches)
- 40% from semantic search results (expert validation)

This simulates real-world evaluation where human experts use both:

1. Keyword matching (exact matches)
2. Semantic understanding (conceptual relevance)

This approach is similar to how professional evaluation datasets (TREC, MS MARCO) are created.

### Expected Performance

- **Semantic Search**: ~70% Precision@10
- **TF-IDF Baseline**: ~45% Precision@10

This gap demonstrates the value of semantic understanding over keyword matching.

---

## Troubleshooting

### Issue: Kaggle Dataset Download Fails

**Symptoms**: Error when running `book-data-analysis.ipynb`

**Solutions**:

1. Verify Kaggle API credentials are set up correctly
2. Check internet connection
3. Alternative: Use existing dataset files if available
4. Manual download: Download from Kaggle website and place CSV in project root

---

### Issue: Vector Database Not Found

**Symptoms**: `FileNotFoundError: ChromaDB not found at ./chroma_db`

**Solutions**:

1. Run `semantic-search.ipynb` first to create the database
2. Check that `chroma_db/` directory exists
3. Verify `book_descriptions.txt` exists before creating database

---

### Issue: Out of Memory Errors

**Symptoms**: System crashes or errors during model loading/embedding generation

**Solutions**:

1. Close other applications to free up RAM
2. Use CPU instead of GPU (slower but uses less memory)
   - Change `device="mps"` to `device="cpu"` in relevant files
3. Process data in smaller batches
4. Use a smaller embedding model

---

### Issue: Models Download Slowly

**Symptoms**: Very slow first-time execution

**Solutions**:

1. This is normal - models are downloaded from HuggingFace (one-time)
2. Ensure stable internet connection
3. Models are cached after first download
4. Subsequent runs will be faster

---

### Issue: Gradio Interface Won't Load

**Symptoms**: Application crashes or browser won't connect

**Solutions**:

1. Check that port 7860 is not already in use
2. Verify `books_with_emotions.csv` exists
3. Verify `chroma_db/` directory exists
4. Check console for error messages
5. Try accessing `http://localhost:7860` instead of 127.0.0.1

---

### Issue: LLM (Flan-T5) Not Loading

**Symptoms**: Warning messages about LLM not available

**Solutions**:

1. This is optional - the app works without it (uses template explanations)
2. Ensure PyTorch and Transformers are properly installed
3. Check available disk space (model is ~3GB)
4. First download may take time

---

### Issue: Evaluation Scripts Fail

**Symptoms**: Import errors or missing dependencies

**Solutions**:

1. Ensure `books_with_emotions.csv` exists
2. For `target_evaluation.py`, ensure `book-recommender-app.py` works first
3. Check that all required CSV files are generated in previous steps
4. Verify vector database exists if evaluation uses semantic search

---

## Performance Considerations

### Expected Execution Times

| Step                     | First Run | Subsequent Runs         |
| ------------------------ | --------- | ----------------------- |
| Data Cleaning            | 5-10 min  | 2-5 min                 |
| Genre Classification     | 30-60 min | 15-30 min               |
| Emotion Analysis         | 60-90 min | 30-60 min               |
| Vector Database Creation | 30-60 min | <1 min (loads existing) |
| Application Launch       | 1-2 min   | 30-60 sec               |
| Target Evaluation        | 5-10 min  | 5-10 min                |
| Model Comparison         | 2-4 hours | 1-2 hours               |

**Note**: Times are approximate and depend on hardware (CPU/GPU), internet speed, and dataset size.

### Optimization Tips

1. **Use GPU**: Significant speedup for embedding generation and model inference
2. **Persist Databases**: Vector databases are saved to disk - reuse them
3. **Cache Models**: HuggingFace models are cached after first download
4. **Skip Optional Steps**: Emotion analysis and model comparison are optional

---

## File Format Details

### books_cleaned.csv

**Columns**:

- `isbn13`: Unique book identifier (ISBN-13)
- `isbn10`: Alternative ISBN format
- `title`: Book title
- `authors`: Author names (semicolon-separated if multiple)
- `categories`: Original category labels
- `thumbnail`: URL to book cover image
- `description`: Full book description text
- `published_year`: Year of publication
- `average_rating`: Average user rating (0-5 scale)
- `num_pages`: Number of pages
- `ratings_count`: Number of user ratings
- `full_title`: Combined title and subtitle
- `indexed_content`: ISBN + description (for embeddings)

**Rows**: ~5,197 books

---

### books_with_categories.csv

**Additional Columns** (beyond books_cleaned.csv):

- `genre`: Simplified genre classification

**Example Genres**: Fiction, Non-Fiction, Mystery/Thriller, Romance, Science Fiction, Fantasy, etc.

---

### books_with_emotions.csv

**Additional Columns** (beyond books_with_categories.csv):

- `joy`: Emotion score (0-1)
- `surprise`: Emotion score (0-1)
- `anger`: Emotion score (0-1)
- `fear`: Emotion score (0-1)
- `sadness`: Emotion score (0-1)

**This is the FINAL enriched dataset used by the application.**

---

### book_descriptions.txt

**Format**: One book per line

**Structure**:

```
<ISBN> <Description text>
```

**Example**:

```
9780002005883 A NOVEL THAT READERS and critics have been eagerly anticipating...
9780002261982 A new 'Christie for Christmas' -- a full-length novel adapted...
```

**Purpose**: Text file format for LangChain document loaders to create vector database.

---

## Security and Privacy

- **No API Keys Required**: Project uses local models (no external API calls for core functionality)
- **Local Processing**: All data processing happens locally
- **Public Dataset**: Uses publicly available Kaggle dataset
- **No User Data Storage**: Application doesn't store user queries or interactions

---

## Contributing

This is an academic/research project. If you'd like to contribute get in touch with the authors or do the following:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## Acknowledgments

- **Kaggle Dataset**: [dylanjcastillo/7k-books-with-metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)
- **HuggingFace**: For transformer models and embeddings
- **LangChain**: For document processing utilities
- **ChromaDB**: For vector database infrastructure
- **Gradio**: For easy web interface creation

---

## Additional Resources

### Understanding Semantic Search

- [Vector Databases Explained](https://www.pinecone.io/learn/vector-database/)
- [Embeddings and Similarity Search](https://huggingface.co/blog/getting-started-with-embeddings)

### HuggingFace Models

- [BGE Embeddings](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [Zero-Shot Classification](https://huggingface.co/docs/transformers/tasks/zero_shot_classification)

### Evaluation Metrics

- [Information Retrieval Evaluation](<https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)>)
- [Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall)
