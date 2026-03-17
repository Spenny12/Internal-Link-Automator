import streamlit as st
import advertools as adv
import pandas as pd
import tempfile
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# --- Configuration & Setup ---
st.set_page_config(page_title="Internal Link Automator", layout="wide")
st.title("Automated Internal Linking Tool")

# --- UI Inputs ---
with st.sidebar:
    st.header("Settings")
    gemini_api_key = st.text_input("Gemini API Key", type="password")
    domain = st.text_input("Domain Name", placeholder="https://example.com")
    
    st.subheader("Target Pages (Incoming Links)")
    target_urls_input = st.text_area("Paste URLs (one per line)", height=150, key="target")
    
    st.subheader("Source Pages (Outgoing Links)")
    source_urls_input = st.text_area("Paste URLs (one per line)", height=150, key="source")
    
    allow_new_copy = st.checkbox("Allow suggestions with NEW copy (1 line)")
    
    # No emojis in the button per requirements
    run_button = st.button("Generate Link Suggestions")

# --- Helper Functions ---
def parse_urls(text_input):
    """Parses newline-separated URLs into a clean list."""
    return [url.strip() for url in text_input.split('\n') if url.strip()]

def crawl_urls(urls):
    """Crawls a list of URLs using advertools and returns a DataFrame."""
    with tempfile.NamedTemporaryFile(suffix='.jl', delete=False) as tmp_file:
        filepath = tmp_file.name
        
    try:
        adv.crawl(urls, filepath, custom_settings={'LOGLEVEL': 'ERROR'})
        df = pd.read_json(filepath, lines=True)
        return df
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

def get_gemini_suggestions(target_url, source_url, source_text, allow_new_copy):
    """Calls Gemini API to get internal linking recommendations."""
    genai.configure(api_key=gemini_api_key)
    # Using the standard text model
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    You are an expert SEO content strategist. 
    Your goal is to find a natural way to add an internal link from the "Source Page" to the "Target Page".
    
    Target Page URL (the page we are linking to): {target_url}
    Source Page URL (the page the link will live on): {source_url}
    
    Source Page Content:
    {source_text[:8000]} 
    
    INSTRUCTIONS:
    1. The anchor text MUST be highly relevant to the Target Page.
    2. Suggest a link using EXISTING copy from the source page. Provide the exact existing sentence and specify which words should be the anchor text.
    """
    
    if allow_new_copy:
        prompt += """
        3. Suggest a link using NEW copy. Write exactly ONE new sentence that fits naturally into the context of the source page, and specify the anchor text.
        """
        
    prompt += """
    Output the response in clean JSON format with the following keys:
    - "existing_copy_sentence": The full sentence from the text.
    - "existing_copy_anchor": The exact words to hyperlink.
    """
    if allow_new_copy:
        prompt += """
    - "new_copy_sentence": The newly written one-line sentence.
    - "new_copy_anchor": The exact words to hyperlink in the new sentence.
    """
    
    prompt += "\nReturn ONLY valid JSON. Do not use Markdown blocks."

    try:
        response = model.generate_content(prompt)
        # Clean up potential markdown formatting from the response
        clean_text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_text)
    except Exception as e:
        return {"error": str(e)}

# --- Main Workflow ---
if run_button:
    if not gemini_api_key:
        st.error("Please provide a Gemini API Key.")
        st.stop()
        
    target_urls = parse_urls(target_urls_input)
    source_urls = parse_urls(source_urls_input)
    
    if not target_urls or not source_urls:
        st.error("Please provide both Target and Source URLs.")
        st.stop()
        
    all_urls = list(set(target_urls + source_urls))
    
    with st.spinner("Crawling pages with Advertools..."):
        crawl_df = crawl_urls(all_urls)
        
        # Ensure we have body text, fallback to title if body is missing
        if 'body_text' not in crawl_df.columns:
            crawl_df['body_text'] = crawl_df.get('title', '')
        
        crawl_df['body_text'] = crawl_df['body_text'].fillna('')
        
        # Create dictionaries mapping URL to text
        url_to_text = dict(zip(crawl_df['url'], crawl_df['body_text']))
        
        # Filter out URLs that failed to crawl or have no text
        valid_targets = [u for u in target_urls if u in url_to_text and url_to_text[u]]
        valid_sources = [u for u in source_urls if u in url_to_text and url_to_text[u]]

    if not valid_targets or not valid_sources:
        st.error("Could not extract text from the provided URLs. Please check the URLs and try again.")
        st.stop()

    with st.spinner("Calculating cosine similarity..."):
        # Combine valid texts to fit the vectorizer
        corpus = [url_to_text[u] for u in valid_targets] + [url_to_text[u] for u in valid_sources]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        target_matrix = tfidf_matrix[:len(valid_targets)]
        source_matrix = tfidf_matrix[len(valid_targets):]
        
        # Calculate similarity (Targets x Sources)
        similarity_matrix = cosine_similarity(target_matrix, source_matrix)

    st.success("Analysis Complete! Generating Suggestions...")
    
    # Process top matches
    for i, target_url in enumerate(valid_targets):
        st.markdown("---")
        st.header(f"Target Page: `{target_url}`")
        
        # Get indices of top 3 similar source pages
        similarities = similarity_matrix[i]
        
        # Create a list of (source_index, similarity_score)
        source_scores = [(j, similarities[j]) for j in range(len(valid_sources))]
        
        # Sort by score descending and filter out the exact same URL
        source_scores.sort(key=lambda x: x[1], reverse=True)
        top_sources = [
            (valid_sources[j], score) for j, score in source_scores 
            if valid_sources[j] != target_url
        ][:3] # Take top 3
        
        if not top_sources:
            st.info("No distinct matching source pages found.")
            continue
            
        for source_url, score in top_sources:
            with st.expander(f"Source Page: {source_url} (Similarity: {score:.2f})", expanded=True):
                with st.spinner("Asking Gemini for link suggestions..."):
                    source_text = url_to_text[source_url]
                    suggestion = get_gemini_suggestions(target_url, source_url, source_text, allow_new_copy)
                    
                    if "error" in suggestion:
                        st.error(f"Error fetching from Gemini: {suggestion['error']}")
                    else:
                        st.subheader("Option 1: Using Existing Copy")
                        st.write(f"**Sentence:** {suggestion.get('existing_copy_sentence', 'N/A')}")
                        st.write(f"**Anchor Text:** `{suggestion.get('existing_copy_anchor', 'N/A')}`")
                        
                        if allow_new_copy and "new_copy_sentence" in suggestion:
                            st.subheader("Option 2: Using New Copy")
                            st.write(f"**Suggested New Line:** {suggestion.get('new_copy_sentence', 'N/A')}")
                            st.write(f"**Anchor Text:** `{suggestion.get('new_copy_anchor', 'N/A')}`")
