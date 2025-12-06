#!/usr/bin/env python3
"""
Book Recommendation System with Semantic Search
A Gradio-based interface for finding books based on semantic similarity
"""

import os
import pandas as pd
import numpy as np
import gradio as gr
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import pipeline
import torch

# Initialize environment variables
load_dotenv()

# Load and preprocess book data
book_dataset = pd.read_csv("books_with_emotions.csv")
book_dataset.loc[:, "large_thumbnail"] = book_dataset["thumbnail"].apply(lambda x: x + "&fife=w800" if pd.notna(x) else "cover-not-found.jpg")

# Initialize embedding model as it is required for loading existing ChromaDB
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5", model_kwargs={"device": "mps"}, encode_kwargs={"normalize_embeddings": True})

# Load existing vector database from disk
chroma_db_path = "./chroma_db"
if os.path.exists(chroma_db_path) and os.listdir(chroma_db_path):
    vector_store = Chroma( persist_directory=chroma_db_path, embedding_function=embeddings)
    print(f"Vector database loaded successfully!")
else:
    raise FileNotFoundError(
        f"ChromaDB not found at {chroma_db_path}. "
        "Please ensure the database exists before running this application."
    )

# Initialize LLM for generating explanations and query understanding
# Using Flan-T5-large model from HuggingFace
print("Loading LLM for explanations and query understanding...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
explanation_llm = None
query_understanding_llm = None

try:
    device_id = 0 if device == "mps" else -1
    explanation_llm = pipeline("text2text-generation", model="google/flan-t5-large", device=device_id)
    # Use the same model for query understanding
    query_understanding_llm = explanation_llm
    print("Loaded Flan-T5-large model successfully!")
except Exception as e:
    print(f"Could not load Flan-T5-large model: {e}")
    explanation_llm = None
    query_understanding_llm = None
    print("Warning: LLM not available. Explanations and query understanding will use fallback methods.")

# Emotion-based sorting configuration
emotion_sort_map = {
    "Happy": "joy",
    "Surprising": "surprise",
    "Angry": "anger",
    "Suspenseful": "fear",
    "Sad": "sadness"
}


def understand_query(user_query: str) -> dict:
    """
    Analyze and understand the user's query to extract intent, themes, and improve search.
    
    Args:
        user_query: The original user query
        
    Returns:
        Dictionary containing:
            - original_query: The original query
            - enhanced_query: Expanded/rewritten query for better semantic search
            - key_themes: List of key themes/concepts extracted
            - intent_summary: Brief summary of what the user is looking for
    """
    if not user_query or not user_query.strip():
        return {
            "original_query": user_query,
            "enhanced_query": user_query,
            "key_themes": [],
            "intent_summary": ""
        }
    
    # If LLM is not available, return original query with basic processing
    if query_understanding_llm is None:
        # Basic keyword extraction and query enhancement
        words = user_query.lower().split()
        # Remove common stop words (basic list)
        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                     'could', 'may', 'might', 'must', 'can', 'about', 'for', 'to', 'of',
                     'in', 'on', 'at', 'by', 'with', 'from', 'as', 'and', 'or', 'but'}
        key_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        return {
            "original_query": user_query,
            "enhanced_query": user_query,  
            "key_themes": key_words[:5],
            "intent_summary": f"Looking for books related to: {', '.join(key_words[:15])}"
        }
    
    try:
        # Step 1: Extract key themes and intent
        intent_prompt = f"""Analyze this book search query and extract key information.

User Query: "{user_query}"

Extract:
1. Main themes or topics (2-4 key words/phrases)
2. What type of book the user wants (fiction/nonfiction, genre hints)
3. Any specific requirements (emotional tone, setting, character types)

Format your response as:
Themes: [list themes]
Type: [fiction/nonfiction/general]
Requirements: [any specific requirements]

Analysis:"""
        
        intent_result = query_understanding_llm(
            intent_prompt,
            max_length=450,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )
        intent_analysis = intent_result[0]["generated_text"].strip()
        
        # Extract themes from analysis
        themes = []
        if "Themes:" in intent_analysis:
            theme_part = intent_analysis.split("Themes:")[1].split("Type:")[0].strip()
            themes = [t.strip() for t in theme_part.replace("[", "").replace("]", "").split(",") if t.strip()]
        
        # Step 2: Expand/rewrite query for better semantic search
        expansion_prompt = f"""Rewrite and expand this book search query to improve semantic search results.
Include synonyms, related concepts, and alternative phrasings that capture the same intent.

Original Query: "{user_query}"

Create an enhanced query that:
- Includes synonyms and related terms
- Maintains the core intent
- Is optimized for finding semantically similar book descriptions
- Is concise (1-2 sentences maximum)

Enhanced Query:"""
        
        expansion_result = query_understanding_llm(
            expansion_prompt,
            max_length=250,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.4, 
            top_p=0.9,
        )
        enhanced_query = expansion_result[0]["generated_text"].strip()
        
        # Clean up the enhanced query
        if "Enhanced Query:" in enhanced_query:
            enhanced_query = enhanced_query.split("Enhanced Query:")[-1].strip()
        # Remove quotes if present
        enhanced_query = enhanced_query.strip('"').strip("'")
        
        # Step 3: Create intent summary
        intent_summary = intent_analysis[:1500] if len(intent_analysis) > 1500 else intent_analysis
        
        return {
            "original_query": user_query,
            "enhanced_query": enhanced_query if enhanced_query else user_query,
            "key_themes": themes if themes else [],
            "intent_summary": intent_summary
        }
        
    except Exception as e:
        print(f"Error in query understanding: {e}")
        # Fallback methof
        return {
            "original_query": user_query,
            "enhanced_query": user_query,
            "key_themes": [],
            "intent_summary": ""
        }


def find_similar_books(
    search_text: str,
    genre_filter: str = None,
    emotion_preference: str = None,
    candidate_count: int = 50,
    result_count: int = 10,
    return_scores: bool = False,
) -> tuple:
    """
    Find books similar to the search query with optional filtering and query understanding
    
    Args:
        search_text: The search query
        genre_filter: Optional genre filter
        emotion_preference: Optional emotion preference
        candidate_count: Number of candidates to retrieve
        result_count: Number of results to return
        return_scores: Whether to return similarity scores
    
    Returns:
        If return_scores=False: DataFrame with matched books
        If return_scores=True: (DataFrame, dict) where dict maps ISBN to similarity score
    """
    # Apply query understanding to enhance the query
    query_info = understand_query(search_text)
    enhanced_query = query_info["enhanced_query"]
    # Use enhanced query for search, but keep original for display
    search_query_for_embedding = enhanced_query
    
    # Get results with similarity scores using the enhanced query
    similar_results_with_scores = vector_store.similarity_search_with_score(search_query_for_embedding, k=candidate_count)
    
    # Extract ISBNs and scores
    isbn_to_score = {}
    isbn_codes = []
    for doc, score in similar_results_with_scores:
        isbn_str = doc.page_content.strip('"').split()[0]
        try:
            isbn_int = int(isbn_str)
            isbn_codes.append(isbn_int)
            # Convert distance to similarity to get higher similarity, as it is the distance between the query and the book
            # ChromaDB returns distance, so we convert it
            similarity_score = 1.0 / (1.0 + score) 
            isbn_to_score[isbn_int] = similarity_score
        except ValueError:
            continue
    
    matched_books = book_dataset[book_dataset["isbn13"].isin(isbn_codes)].copy()
    
    # Add similarity score column to the dataframe
    matched_books["similarity_score"] = matched_books["isbn13"].map(isbn_to_score).fillna(0.0)
    
    # Sort by similarity score in descending order (highest match first)
    matched_books = matched_books.sort_values(by="similarity_score", ascending=False)

    # Apply genre filter if specified
    if genre_filter and genre_filter != "All":
        matched_books = matched_books[matched_books["genre"] == genre_filter].head(result_count)
    else:
        matched_books = matched_books.head(result_count)

    # Apply emotion-based sorting if specified (secondary sort after similarity)
    if emotion_preference and emotion_preference != "All" and emotion_preference in emotion_sort_map:
        sort_column = emotion_sort_map[emotion_preference]
        # Sort by emotion score, but maintain similarity score as primary sort
        matched_books = matched_books.sort_values(by=[sort_column, "similarity_score"], ascending=[False, False])

    if return_scores:
        return matched_books, isbn_to_score
    return matched_books


def format_author_names(author_string: str) -> str:
    """
    Format author names for display
    """
    author_list = author_string.split(";")
    author_count = len(author_list)
    
    if author_count == 1:
        return author_list[0]
    elif author_count == 2:
        return f"{author_list[0]} and {author_list[1]}"
    else:
        return f"{', '.join(author_list[:-1])}, and {author_list[-1]}"


def _generate_template_explanation(
    similarity_score: float,
    genre: str = None,
    emotion_preference: str = None,
    emotion_scores: dict = None,
) -> str:
    """
    Generate a template-based explanation (fallback when LLM is not available)
    """
    explanation_parts = []
    explanation_parts.append(f"This book matches your query with a similarity score of {similarity_score:.2%}.")
    if genre and genre != "All":
        explanation_parts.append(f"It's classified as {genre}.")
    if emotion_preference and emotion_preference != "All" and emotion_scores:
        emotion_key = emotion_sort_map.get(emotion_preference, "").lower()
        if emotion_key in emotion_scores:
            score = emotion_scores[emotion_key]
            explanation_parts.append(f"It has a {emotion_preference.lower()} tone (score: {score:.2f}).")
    return " ".join(explanation_parts)


def generate_explanation(
    user_query: str,
    book_title: str,
    authors: str,
    description: str,
    genre: str,
    similarity_score: float,
    emotion_scores: dict = None,
    emotion_preference: str = None,
) -> str:
    """
    Generate an explanation for why a book matches the user's query using LLM
    """
    if explanation_llm is None:
        # Fallback template-based explanation if LLM is not available
        return _generate_template_explanation(similarity_score, genre, emotion_preference, emotion_scores)
    
    # Build the prompt for the LLM
    formatted_authors = format_author_names(authors)
    description_preview = description[:800] + "..." if len(description) > 800 else description
    
    # Format emotion information
    emotion_info = ""
    if emotion_scores:
        top_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        emotion_info = f"Emotional tone: {', '.join([f'{k} ({v:.2f})' for k, v in top_emotions])}."
        if emotion_preference and emotion_preference != "All":
            emotion_key = emotion_sort_map.get(emotion_preference, "").lower()
            if emotion_key in emotion_scores:
                emotion_info += f" User requested {emotion_preference.lower()} tone."
    
    # Create detailed prompt for Flan-T5 model to generate comprehensive explanations
    prompt = f"""Write a detailed explanation of why this book matches the user's query. Be specific and comprehensive.

User's Query: "{user_query}"

Book Information:
- Title: {book_title}
- Author: {formatted_authors}
- Genre: {genre if genre else "Unknown"}
- Semantic Similarity Score: {similarity_score:.2%} (higher means better match)
{emotion_info if emotion_info else ""}

Instructions:
1. Explain how the book's content, themes, or style relate to what the user is looking for
2. Mention the semantic similarity score and what it indicates about the match quality
3. Discuss how the genre aligns with the user's query (if relevant)
4. If emotion information is provided, explain how the book's emotional tone matches the user's preferences
5. Be specific about which aspects of the book make it a good recommendation
6. Write multiple sentences that are clear, informative, and helpful

Detailed Explanation:"""
    
    try:
        # Generated explanation using Flan-T5 for verbose output
        result = explanation_llm(prompt,max_length=250,  num_return_sequences=1,do_sample=True,temperature=0.3, top_p=0.9)
        explanation = result[0]["generated_text"].strip()
        
        # Clean up the explanation if it contains the prompt text
        if "Detailed Explanation:" in explanation:
            explanation = explanation.split("Detailed Explanation:")[-1].strip()
        
        # Fallback if explanation is too short or empty
        if len(explanation) < 20:
            raise ValueError("Explanation too short")
        
        return explanation
        
    except Exception as e:
        print(f"Error generating explanation: {e}")
        # Return template-based fallback
        return _generate_template_explanation(similarity_score, genre, emotion_preference, emotion_scores)


def generate_recommendations(
    user_input: str,
    selected_genre: str,
    selected_emotion: str,
):
    """
    Generate and format book recommendations for display with AI explanations
    
    Args:
        user_input: The user's search query
        selected_genre: Selected genre filter
        selected_emotion: Selected emotion preference
    """
    # Get books with similarity scores
    matched_books_df, isbn_to_score = find_similar_books(
        user_input, selected_genre, selected_emotion, 
        return_scores=True
    )
    
    formatted_results = []

    for idx in matched_books_df.index:
        book_entry = matched_books_df.loc[idx]
        book_isbn = book_entry["isbn13"]
        book_description = str(book_entry["description"]) if pd.notna(book_entry["description"]) else "No description available."
        description_words = book_description.split()
        # Use 40 words for better context
        short_description = " ".join(description_words[:40]) + "..." if len(description_words) > 40 else book_description

        formatted_authors = format_author_names(book_entry["authors"])
        
        # Get similarity score
        similarity_score = isbn_to_score.get(book_isbn, 0.0)
        
        # Get emotion scores if available
        emotion_scores = {}
        if pd.notna(book_entry.get("joy")):
            emotion_scores = {
                "joy": book_entry.get("joy", 0),
                "surprise": book_entry.get("surprise", 0),
                "anger": book_entry.get("anger", 0),
                "fear": book_entry.get("fear", 0),
                "sadness": book_entry.get("sadness", 0),
            }
        
        # Generate explanation
        explanation = ""
        if user_input.strip():
            explanation = generate_explanation(
                user_query=user_input,
                book_title=book_entry["title"],
                authors=book_entry["authors"],
                description=book_description,
                genre=book_entry.get("genre", ""),
                similarity_score=similarity_score,
                emotion_scores=emotion_scores if emotion_scores else None,
                emotion_preference=selected_emotion,
            )
        
        # Format display text with clear sections and better structure
        title = book_entry['title']
        
        # Build metadata section with better visual separation
        metadata_items = []
        
        # Author
        metadata_items.append(f"👤 **Author:** {formatted_authors}")
        
        # Genre
        if book_entry.get('genre') and pd.notna(book_entry.get('genre')):
            metadata_items.append(f"📚 **Genre:** {book_entry.get('genre')}")
        
        # Published Year
        if pd.notna(book_entry.get('published_year')):
            year = int(book_entry.get('published_year'))
            metadata_items.append(f"📅 **Published:** {year}")
        
        # Number of Pages
        if pd.notna(book_entry.get('num_pages')):
            pages = int(book_entry.get('num_pages'))
            metadata_items.append(f"**Pages:** {pages:,}")
        
        # Rating
        if pd.notna(book_entry.get('average_rating')):
            rating = book_entry.get('average_rating')
            ratings_count = ""
            if pd.notna(book_entry.get('ratings_count')):
                count = int(book_entry.get('ratings_count'))
                ratings_count = f" ({count:,} ratings)"
            metadata_items.append(f"⭐ **Rating:** {rating:.1f}/5.0{ratings_count}")
        
        # Similarity Score
        metadata_items.append(f"🎯 **Match Score:** {similarity_score:.1%}")
        
        # Join metadata with double line breaks for better spacing
        metadata_section = "\n\n".join(metadata_items)
        
        # Build the complete display text with clear sections and better spacing
        if explanation:
            display_text = f"""# {title}

---

## 📋 Book Details

{metadata_section}

---

## 📖 Description

{short_description}

---

## 💡 Why This Book Matches Your Query

{explanation}

---"""
        else:
            display_text = f"""# {title}

---

## 📋 Book Details

{metadata_section}

---

## 📖 Description

{short_description}

---"""
        
        formatted_results.append((book_entry["large_thumbnail"], display_text))
    
    return formatted_results


# Prepare UI options
available_genres = ["All"] + sorted(book_dataset["genre"].dropna().unique().tolist())
available_emotions = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Build Gradio interface with improved UI
app_interface = gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
    ),
    css="""
    .book-gallery {
        gap: 25px !important;
        padding: 20px 0 !important;
        max-height: 85vh !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        scrollbar-width: thin !important;
        scrollbar-color: #c0c0c0 #f0f0f0 !important;
    }
    .book-gallery::-webkit-scrollbar {
        width: 10px !important;
    }
    .book-gallery::-webkit-scrollbar-track {
        background: #2a2a2a !important;
        border-radius: 5px !important;
    }
    .book-gallery::-webkit-scrollbar-thumb {
        background: #5a5a5a !important;
        border-radius: 5px !important;
    }
    .book-gallery::-webkit-scrollbar-thumb:hover {
        background: #7a7a7a !important;
    }
    .book-gallery .gallery-item {
        border-radius: 12px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
        padding: 25px !important;
        background: #1e1e1e !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
        border: 1px solid #3a3a3a !important;
        max-width: 100% !important;
        min-height: 650px !important;
        max-height: 90vh !important;
        display: flex !important;
        flex-direction: column !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }
    .book-gallery .gallery-item:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.5) !important;
        border-color: #4a90e2 !important;
        background: #252525 !important;
    }
    .book-gallery img {
        border-radius: 8px !important;
        margin-bottom: 15px !important;
        width: 100% !important;
        height: auto !important;
        max-height: 300px !important;
        object-fit: contain !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
        flex-shrink: 0 !important;
    }
    .book-gallery .caption {
        font-size: 16px !important;
        line-height: 2.2 !important;
        color: #ffffff !important;
        padding: 20px 15px !important;
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        white-space: pre-wrap !important;
        text-align: left !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
        overflow-y: auto !important;
        max-height: 100% !important;
    }
    .book-gallery .caption h1 {
        color: #ffffff !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        margin-bottom: 15px !important;
        margin-top: 0 !important;
        border-bottom: 3px solid #4a90e2 !important;
        padding-bottom: 12px !important;
        line-height: 1.3 !important;
    }
    .book-gallery .caption h2 {
        color: #e0e0e0 !important;
        font-size: 20px !important;
        font-weight: 600 !important;
        margin-top: 20px !important;
        margin-bottom: 12px !important;
        padding-top: 10px !important;
        border-top: 2px solid #4a4a4a !important;
        line-height: 1.4 !important;
    }
    .book-gallery .caption p {
        font-size: 16px !important;
        line-height: 1.8 !important;
        margin: 8px 0 !important;
        color: #e0e0e0 !important;
    }
    .book-gallery .caption strong {
        color: #ffffff !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        display: inline-block !important;
        min-width: 120px !important;
    }
    .book-gallery .caption em {
        color: #b0b0b0 !important;
        font-style: italic !important;
        font-size: 15px !important;
    }
    .book-gallery .caption hr {
        border: none !important;
        border-top: 2px solid #4a4a4a !important;
        margin: 20px 0 !important;
    }
    .header {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .footer {
        margin-top: 30px;
    }
    """
)

with app_interface:
    # Header
    gr.Markdown(
        """
        # 📚 Semantic Book Recommender
        ### AI-Powered Book Recommendations with Explanations
        Find your next great read based on natural language descriptions, with AI-generated explanations for each recommendation.
        """,
        elem_classes="header"
    )
    
    # Search section
    with gr.Row():
        search_input = gr.Textbox(
            label="🔍 What kind of book are you looking for?",
            placeholder="e.g., A story about forgiveness and redemption, A thrilling mystery with unexpected twists, A book to teach children about nature...",
            lines=3,
            info="Describe the book you want to find in natural language"
        )
    
    # Filters
    with gr.Row():
        genre_selector = gr.Dropdown(
            choices=available_genres,
            label="📖 Genre",
            value="All",
            info="Filter by book category"
        )
        emotion_selector = gr.Dropdown(
            choices=available_emotions,
            label="😊 Emotional Tone",
            value="All",
            info="Prefer books with a specific emotional tone"
        )
        search_button = gr.Button(
            "🚀 Find Recommendations",
            variant="primary",
            size="lg",
            scale=1
        )
    
    gr.Markdown("---")
    
    # Results section
    gr.Markdown("## 📋 Recommendations")
    status_msg = gr.Markdown("", visible=False)
    with gr.Row():
        results_gallery = gr.Gallery(
            label="",
            columns=2,
            rows=5,
            height=800,
            show_label=False,
            elem_classes="book-gallery",
            allow_preview=True,
            show_download_button=False
        )
    
    # Footer info
    gr.Markdown(
        """
        ---
        <div style='text-align: center; color: #666; font-size: 0.9em; padding: 20px;'>
        <p>💡 <strong>Tip:</strong> Be specific in your query for better results. Try describing themes, settings, character types, or the mood you're looking for.</p>
        </div>
        """,
        elem_classes="footer"
    )
    
    def generate_with_status(user_input, genre, emotion):
        """Wrapper to show status during generation"""
        if not user_input.strip():
            return gr.update(visible=True, value="⚠️ Please enter a search query."), []
        
        try:
            results = generate_recommendations(
                user_input, genre, emotion
            )
            return gr.update(visible=False), results
        except Exception as e:
            return gr.update(visible=True, value=f"❌ Error: {str(e)}"), []
    
    search_button.click(
        fn=generate_with_status,
        inputs=[search_input, genre_selector, emotion_selector],
        outputs=[status_msg, results_gallery]
    )


if __name__ == "__main__":
    app_interface.launch()