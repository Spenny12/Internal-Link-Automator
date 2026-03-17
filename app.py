import streamlit as st
import advertools as adv
import pandas as pd
import tempfile
import json
import os
import concurrent.futures
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
    model = genai.GenerativeModel('gemini-3-flash-preview')

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

        if 'body_text' not in crawl_df.columns:
            crawl_df['body_text'] = crawl_df.get('title', '')

        crawl_df['body_text'] = crawl_df['body_text'].fillna('')
        url_to_text = dict(zip(crawl_df['url'], crawl_df['body_text']))

        valid_targets = [u for u in target_urls if u in url_to_text and url_to_text[u]]
        valid_sources = [u for u in source_urls if u in url_to_text and url_to_text[u]]

    if not valid_targets or not valid_sources:
        st.error("Could not extract text from the provided URLs. Please check the URLs and try again.")
        st.stop()

    with st.spinner("Calculating cosine similarity..."):
        corpus = [url_to_text[u] for u in valid_targets] + [url_to_text[u] for u in valid_sources]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(corpus)

        target_matrix = tfidf_matrix[:len(valid_targets)]
        source_matrix = tfidf_matrix[len(valid_targets):]
        similarity_matrix = cosine_similarity(target_matrix, source_matrix)

    st.success("Analysis Complete! Generating Suggestions Concurrently...")

    # --- Prepare Tasks for Batching/Concurrency ---
    tasks = []
    for i, target_url in enumerate(valid_targets):
        similarities = similarity_matrix[i]
        source_scores = [(j, similarities[j]) for j in range(len(valid_sources))]
        source_scores.sort(key=lambda x: x[1], reverse=True)

        top_sources = [
            (valid_sources[j], score) for j, score in source_scores
            if valid_sources[j] != target_url
        ][:3]

        for source_url, score in top_sources:
            tasks.append({
                'target_url': target_url,
                'source_url': source_url,
                'source_text': url_to_text[source_url],
                'score': score
            })

    # --- Execute Tasks Concurrently ---
    results_map = {}
    progress_bar = st.progress(0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_task = {
            executor.submit(
                get_gemini_suggestions,
                task['target_url'],
                task['source_url'],
                task['source_text'],
                allow_new_copy
            ): task for task in tasks
        }

        completed = 0
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            completed += 1
            progress_bar.progress(completed / len(tasks))

            try:
                suggestion = future.result()
            except Exception as e:
                suggestion = {"error": str(e)}

            # Store results mapped by target_url to render in order later
            if task['target_url'] not in results_map:
                results_map[task['target_url']] = []

            results_map[task['target_url']].append({
                'source_url': task['source_url'],
                'score': task['score'],
                'suggestion': suggestion
            })

    # --- Render UI and Prepare CSV Data ---
    csv_data = []

    for target_url in valid_targets:
        if target_url not in results_map:
            continue

        st.markdown("---")
        st.header(f"Target Page: `{target_url}`")

        # Sort back by score since concurrent futures finish out of order
        mapped_sources = sorted(results_map[target_url], key=lambda x: x['score'], reverse=True)

        for item in mapped_sources:
            source_url = item['source_url']
            score = item['score']
            suggestion = item['suggestion']

            # Prepare formatted string for the CSV output
            formatted_suggestion = ""

            with st.expander(f"Source Page: {source_url} (Similarity: {score:.2f})", expanded=True):
                if "error" in suggestion:
                    st.error(f"Error fetching from Gemini: {suggestion['error']}")
                    formatted_suggestion = f"Error: {suggestion['error']}"
                else:
                    st.subheader("Option 1: Using Existing Copy")
                    existing_sentence = suggestion.get('existing_copy_sentence', 'N/A')
                    existing_anchor = suggestion.get('existing_copy_anchor', 'N/A')
                    st.write(f"**Sentence:** {existing_sentence}")
                    st.write(f"**Anchor Text:** `{existing_anchor}`")

                    formatted_suggestion += f"EXISTING COPY\nSentence: {existing_sentence}\nAnchor: {existing_anchor}"

                    if allow_new_copy and "new_copy_sentence" in suggestion:
                        st.subheader("Option 2: Using New Copy")
                        new_sentence = suggestion.get('new_copy_sentence', 'N/A')
                        new_anchor = suggestion.get('new_copy_anchor', 'N/A')
                        st.write(f"**Suggested New Line:** {new_sentence}")
                        st.write(f"**Anchor Text:** `{new_anchor}`")

                        formatted_suggestion += f"\n\nNEW COPY\nSentence: {new_sentence}\nAnchor: {new_anchor}"

            # Append to CSV payload
            csv_data.append({
                "Origin page": source_url,
                "Destination page": target_url,
                "suggested link": formatted_suggestion
            })

    # --- Export to CSV ---
    if csv_data:
        st.markdown("---")
        st.subheader("Export Results")
        export_df = pd.DataFrame(csv_data)

        st.download_button(
            label="Download CSV Export",
            data=export_df.to_csv(index=False).encode('utf-8'),
            file_name="internal_linking_suggestions.csv",
            mime="text/csv"
        )
