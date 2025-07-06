import os
from collections import Counter
from google.cloud import language_v1
from loguru import logger
from bs4 import BeautifulSoup
import math
import re
from typing import List, Set, Dict, Any
from langchain_openai import ChatOpenAI
import aiohttp
from aiohttp import ClientSession
from data_models import EntityRecommendations, EntitySelection, EntitySelections
from dotenv import load_dotenv

load_dotenv()

# Common English stop words
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
    'will', 'with', 'the', 'this', 'but', 'they', 'have', 'had', 'what', 'when',
    'where', 'who', 'which', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'can', 'my', 'your', 'i', 'you', 'we'
}

# Initialize LLM
model = ChatOpenAI(model="gpt-4o")

def tokenize(text: str) -> List[str]:
    """Split text into words, removing punctuation and converting to lowercase."""
    # Remove special characters and convert to lowercase
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Split on whitespace and filter out empty strings
    return [word for word in text.split() if word]

def get_ngrams(words: List[str], n: int = 2, min_frequency: int = 1) -> Dict[str, int]:
    """Generate n-grams from a list of words with frequency filtering."""
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i + n])
        ngrams.append(ngram)

    # Count frequencies and filter
    ngram_counts = Counter(ngrams)
    return {ngram: count for ngram, count in ngram_counts.items() if count >= min_frequency}

def remove_stopwords(words: List[str]) -> List[str]:
    """Remove stop words from a list of words."""
    return [word for word in words if word not in STOP_WORDS]

def calculate_tf_idf(term, document, all_documents):
    """Calculates TF-IDF for a term in a document."""
    term_count = document.lower().split().count(term.lower())
    if term_count == 0:
        return 0
    tf = term_count / len(document.lower().split())
    
    document_count = sum(1 for doc in all_documents if term.lower() in doc.lower())
    if document_count == 0:
        return 0
    idf = math.log(len(all_documents) / document_count)
    return tf * idf

async def scrape_content(url: str, session: ClientSession):
    """Scrapes content from a URL."""
    try:
        async with session.get(url, timeout=10) as response:
            response.raise_for_status()  # Raise an exception for bad status codes
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            # Remove script and style tags
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator=' ', strip=True)
            return text
    except aiohttp.ClientError as e:
        print(f"Error scraping {url}: {e}")
        return None

async def analyze_content(content, credentials_path, all_documents):
    """Analyzes content using Google Cloud Natural Language API."""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    async with language_v1.LanguageServiceAsyncClient() as client:
        type_ = language_v1.Document.Type.PLAIN_TEXT
        document = {"content": content, "type_": type_}
        encoding_type = language_v1.EncodingType.UTF8

        # Analyze Entities
        response = await client.analyze_entities(
            request={"document": document, "encoding_type": encoding_type}
        )

        # Analyze Sentiment
        sentiment_response = await client.analyze_sentiment(
            request={"document": document, "encoding_type": encoding_type}
        )

        entities = {}
        for entity in response.entities:
            entities[entity.name] = {
                "type": language_v1.Entity.Type(entity.type_).name,
                "salience": entity.salience,
                "mentions": [],
                "sentiment": {
                    "score": entity.sentiment.score,
                    "magnitude": entity.sentiment.magnitude
                }
            }
            for mention in entity.mentions:
                entities[entity.name]["mentions"].append({
                    "text": mention.text.content,
                    "type": language_v1.EntityMention.Type(mention.type_).name,
                    "begin_offset": mention.text.begin_offset
                })

        # Basic Keyword Analysis
        words = tokenize(content)
        words = remove_stopwords(words)
        word_counts = Counter(words)
        total_words = len(words)

        # Phrase Extraction
        phrases = get_ngrams(words)
        phrase_counts = Counter(phrases)

        keyword_analysis = {}
        for entity_name, entity_data in entities.items():
            if entity_name in word_counts:
                keyword_analysis[entity_name] = {
                    "density": (word_counts[entity_name] / total_words) * 100,
                    "count": word_counts[entity_name],
                    "phrase_counts": phrase_counts.get(entity_name, 0),
                    "tf_idf": calculate_tf_idf(entity_name, content, all_documents)
                }
            else:
                keyword_analysis[entity_name] = {
                    "density": 0,
                    "count": 0,
                    "phrase_counts": 0,
                    "tf_idf": 0
                }

        return {
            "entities": entities,
            "document_sentiment": {
                "score": sentiment_response.document_sentiment.score,
                "magnitude": sentiment_response.document_sentiment.magnitude
            },
            "keyword_analysis": keyword_analysis,
        }

def compare_pages(client_analysis, competitive_analyses, competitive_pages):
    """Compares the client page analysis to the competitive pages."""
    client_entities = client_analysis.get("entities", {})
    client_keywords = client_analysis.get("keyword_analysis", {})

    missing_entities = {}
    missing_keywords = {}

    for i, comp_analysis in enumerate(competitive_analyses):
        comp_entities = comp_analysis.get("entities", {})
        comp_keywords = comp_analysis.get("keyword_analysis", {})
        comp_url = competitive_pages[i]

        for entity_name, entity_data in comp_entities.items():
            if entity_name not in client_entities:
                if entity_name not in missing_entities:
                    missing_entities[entity_name] = {
                        "competitors": {comp_url: {"salience": entity_data["salience"]}},
                        "type": entity_data["type"]
                    }
                else:
                    missing_entities[entity_name]["competitors"][comp_url] = {"salience": entity_data["salience"]}

        for keyword, data in comp_keywords.items():
            if keyword not in client_keywords:
                if keyword not in missing_keywords:
                    missing_keywords[keyword] = {
                        "competitors": {comp_url: data},
                    }
                else:
                    missing_keywords[keyword]["competitors"][comp_url] = data

    # Filter missing entities to only include those present in at least 2 competitors
    filtered_missing_entities = {
        entity_name: data
        for entity_name, data in missing_entities.items()
        if len(data["competitors"]) >= 2
    }

    # Filter missing keywords to only include those present in at least 2 competitors
    filtered_missing_keywords = {
        keyword: data
        for keyword, data in missing_keywords.items()
        if len(data["competitors"]) >= 2
    }

    return {
        "missing_entities": filtered_missing_entities,
        "missing_keywords": filtered_missing_keywords,
    }


async def generate_entity_recommendations(entity_item: EntitySelection, client_content: str) -> EntityRecommendations:
    """Generates structured recommendations for integrating a target entity."""
    instructions = """
    You are an expert SEO content strategist. Your task is to analyze a client's webpage content and provide specific, actionable recommendations on how to integrate a target entity effectively.

    **Context:**

    *   **Entities are more than just keywords:** They represent topics, concepts, or ideas. Effective integration requires understanding the entity's meaning, related terms, and user intent.
    *   **Avoid keyword stuffing:** Do not simply repeat the entity's name throughout the content. Focus on natural language and readability.
    *   **Strategic placement is key:** Consider where the entity and related terms will fit naturally within the content (e.g., title, headings, introduction, body, conclusion).
    *   **Provide value to users:** Ensure the content is informative, engaging, and addresses user needs.
    *   **Use related terms:** Incorporate synonyms, LSI keywords, and related concepts to expand the semantic scope.

    **Input:**

    *   **Target Entity Info:** 
        *   **Entity Name**: "{entity_name}"
        *   **Relevance Score**: {relevance_score}
        *   **Reasoning**: {reasoning}
    *   **Client Page Content:**
        ```
        {client_page_content}
        ```

    **Instructions:**

    1.  **Analyze the Client Page Content:** Review the provided content to understand its current focus, structure, and tone.
    2.  **Research the Target Entity:** Understand the meaning of "{entity_name}", its key aspects, related terms, and user intent.
    3.  **Identify Integration Opportunities:** Determine where the entity and related terms can be naturally incorporated into the existing content.
    4.  **Provide Specific Recommendations:**
        *   Suggest specific sections or paragraphs where the entity can be discussed.
        *   Recommend related terms and concepts to use alongside the entity.
        *   Provide examples of how to phrase sentences and paragraphs to incorporate the entity naturally.
        *   Suggest where to place the entity in key areas like the title tag, meta description, and headings.
        *   Explain *why* each recommendation is beneficial for both SEO and user experience.
    5.  **Focus on User Value:** Ensure that the recommendations will result in content that is valuable, informative, and engaging for users.
    6.  **Avoid Keyword Stuffing:** Do not recommend simply repeating the entity's name throughout the content.
    """
    prompt = instructions.format(
        entity_name=entity_item.entity_name, 
        relevance_score=entity_item.relevance_score, 
        reasoning=entity_item.reasoning, 
        client_page_content=client_content
    )
    inputs_for_recommendation = [
        ("system", "Provide a structured recommendation for integrating the target entity."), 
        ("user", prompt)
    ]
    structured_recommendation_model = model.with_structured_output(EntityRecommendations)
    output = await structured_recommendation_model.ainvoke(inputs_for_recommendation)
    logger.info(f'EntityRecommendation: \n\n {output}')
    return  output


async def select_entities_for_integration(missing_entities: Dict[str, Any]) -> EntitySelections:
    """Selects the most relevant entities for integration using AI."""
    if not missing_entities:
        return EntitySelections(selected_entities=[])

    # get array of objects with these keys {"entity_name", "entity_type", "count_of_competitors_with_entity, "max_salience"}
    entities = [{"entity_name": entity_name, "entity_type": data["type"], "count_of_competitors_with_entity": len(data["competitors"]), "max_salience": max([comp["salience"] for comp in data["competitors"].values()]), "competitors": list(data["competitors"].keys())} for entity_name, data in missing_entities.items()]
    structured_string_of_entities = "\n".join([f"- Entity Name: {entity['entity_name']}, Entity Type: {entity['entity_type']}, Count of Competitors with Entity: {entity['count_of_competitors_with_entity']}, Max Salience: {entity['max_salience']}, Competitors: {entity['competitors']}" for entity in entities])
    
    prompt = """
    You are an expert SEO content strategist. Your task is to analyze a list of missing entities and select the top 10 most relevant entities to integrate into a client's webpage.

    **Context:**

    *   **Entities are more than just keywords:** They represent topics, concepts, or ideas. Effective integration requires understanding the entity's meaning, related terms, and user intent.
    *   **Relevance is key:** Select entities that are most relevant to the client's page content and user intent.
    *   **Prioritize entities:** Prioritize entities that are most likely to improve the client's page ranking and user experience.
    *   **Provide a relevance score:** Provide a relevance score between 0 and 1 for each entity.
    *   **Provide reasoning:** Provide reasoning behind the selection of each entity.

    **Input:**

    *   **Missing Entities (includes the entity name, count of competitors with the entity, and maximum salience. use this context to determine relevance):**
        ```
        {entity_details}
        ```

    **Instructions:**

    1.  **Analyze the Missing Entities:** Review the provided list of missing entities to understand their meaning and relevance.
    2.  **Select Top 10 Entities:** Select the top 10 most relevant entities to integrate into the client's webpage.
    3.  **Provide a List of Entities:** Provide a list of the selected entities with their relevance scores and reasoning.
    """
    
    inputs_for_selection = [
        ("system", "Select the top 10 most relevant entities to integrate with relevance scores and reasoning."), 
        ("user", prompt.format(entity_details=structured_string_of_entities))
    ]
    structured_selection_model = model.with_structured_output(EntitySelections)
    output = await structured_selection_model.ainvoke(inputs_for_selection)
    logger.info(f'EntitySelections: \n\n {output}')
    return output