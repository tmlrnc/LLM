import os
import requests
import pdfplumber
import openai
import json
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup

def search_for_articles(query):
    """
    Search for machine learning articles related to healthcare and BERT.
    """
    search_url = f"https://www.google.com/search?q={query}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract URLs from the search results
    links = []
    for item in soup.find_all('a'):
        href = item.get('href')
        if 'url?q=' in href and not 'webcache' in href:
            link = href.split('url?q=')[1].split('&sa=U')[0]
            links.append(link)

    return links[:5]  # Return top 5 links

# Set your OpenAI API key
openai.api_key = "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"

# Initialize the BERT model for topic extraction using a pre-trained SentenceTransformer
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def download_pdf(url, filename):
    """
    Downloads a PDF from a given URL and saves it locally.
    Handles errors that might occur during the download process.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename}: {e}")
        return None  # Return None if download fails
    return filename

def extract_text_from_pdf(filename):
    """
    Extracts text from a PDF using pdfplumber.
    Handles errors that might occur during file access or text extraction.
    """
    try:
        text = ""
        with pdfplumber.open(filename) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        return ""

def extract_topics_with_bert(text):
    """
    Uses a BERT model to extract key topics from the text.
    The function identifies the most relevant sentences based on semantic similarity.
    """
    try:
        sentences = text.split('.')
        embeddings = bert_model.encode(sentences, convert_to_tensor=True)
        topics = util.paraphrase_mining(bert_model, sentences)
        
        # Extract the top-ranked sentences using the indices from paraphrase_mining
        extracted_topics = set()  # Use a set to avoid duplicate sentences
        for score, i, j in topics:
            extracted_topics.add(sentences[i])
            extracted_topics.add(sentences[j])
        
        return list(extracted_topics)
    except Exception as e:
        print(f"Error extracting topics with BERT: {e}")
        return []

def process_text_with_openai(text):
    """
    Sends the extracted text to the OpenAI API for structured data extraction.
    The prompt is designed to extract specific healthcare and machine learning attributes in JSON format.
    """
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Extract the following healthcare and machine learning attributes from the text in a structured JSON format:\n\n"
                   f"Motion: [Extracted]\n"
                   f"Fatigue: [Extracted]\n"
                   f"LSTM: [Extracted]\n"
                   f"BERT sentence_transformers: [Extracted]\n"
                   f"Eye Movement: [Extracted]\n"
                   f"Right_Eye_Openness: [Extracted]\n\n"
                   f"Text: {text}\n\n"
                   f"Provide the extracted information in a JSON format.",
            max_tokens=1500,
            temperature=0.5
        )
        return json.loads(response.choices[0].text.strip())
    except Exception as e:
        print(f"Error processing text with OpenAI: {e}")
        return {}

def validate_extracted_data(data):
    """
    Validates the extracted data to ensure all required fields are present.
    If any fields are missing or empty, a warning is printed.
    """
    required_keys = [
        "Motion", "Fatigue", "LSTM", "BERT sentence_transformers", 
        "Eye Movement", "Right_Eye_Openness"
    ]
    for key in required_keys:
        if key not in data or not data[key]:
            print(f"Warning: Missing or incomplete data for {key}")
    
    return data

def save_json(data, filename):
    """
    Saves the extracted and validated data to a JSON file.
    Handles errors that might occur during file operations.
    """
    try:
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"Saved data to {filename}")
    except Exception as e:
        print(f"Error saving JSON to {filename}: {e}")

def main():
    """
    Main function to orchestrate the process:
    1. Download PDFs from given URLs.
    2. Extract text from the downloaded PDFs.
    3. Use BERT to extract key topics from the text.
    4. Send the topics to the OpenAI API for structured attribute extraction.
    5. Validate the extracted data.
    6. Save the data to a JSON file.
    """
    # Step 1: Search for articles and add URLs to the list
    search_query = "machine learning in healthcare BERT"
    article_links = search_for_articles(search_query)
    
    urls = [
        "https://stgendev01.blob.core.windows.net/python-test/nw1.pdf",
        "https://stgendev01.blob.core.windows.net/python-test/BOV.pdf"
    ] + article_links  # Add article links to the URLs list
    
    filenames = ["nw1.pdf", "BOV.pdf"] + [f"article_{i+1}.pdf" for i in range(len(article_links))]

    for url, filename in zip(urls, filenames):
        # Step 2: Download PDF
        downloaded_file = download_pdf(url, filename)
        if downloaded_file is None:
            continue  # Skip processing if download failed

        # Step 3: Extract text from PDF
        text = extract_text_from_pdf(downloaded_file)
        if not text:
            continue  # Skip processing if text extraction failed

        # Step 4: Extract key topics using BERT
        topics = extract_topics_with_bert(text)
        if not topics:
            continue  # Skip processing if topic extraction failed
        
        # Combine topics into a single text block
        combined_text = " ".join(topics)
        
        # Step 5: Process text with OpenAI to extract attributes
        openai_output = process_text_with_openai(combined_text)
        
        # Step 6: Validate extracted data
        validated_data = validate_extracted_data(openai_output)
        
        # Step 7: Save the data to a JSON file
        output_filename = f"output-{os.path.splitext(filename)[0]}.json"
        save_json(validated_data, output_filename)

if __name__ == "__main__":
    main()
