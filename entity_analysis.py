import os
import asyncio
from collections import Counter
from google.cloud import language_v1
from google.api_core import exceptions as google_exceptions
from loguru import logger
from bs4 import BeautifulSoup
import math
import re
from typing import List, Set, Dict, Any, Optional
from langchain_openai import ChatOpenAI
import aiohttp
from aiohttp import ClientSession, ClientTimeout, ClientError
from data_models import EntityRecommendations, EntitySelection, EntitySelections, AnalysisProgress
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from asyncio_throttle import Throttler
import validators
import streamlit as st

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

# Rate limiting: 1 request per 2 seconds
RATE_LIMITER = Throttler(rate_limit=1, period=2)

# Initialize LLM with error handling
def get_llm_model():
    """Get OpenAI model with proper error handling."""
    try:
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in secrets or environment variables")
        return ChatOpenAI(model="gpt-4o", api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI model: {e}")
        raise

model = get_llm_model()

def tokenize(text: str) -> List[str]:
    """Split text into words, removing punctuation and converting to lowercase."""
    if not text:
        return []
    # Remove special characters and convert to lowercase
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Split on whitespace and filter out empty strings
    return [word for word in text.split() if word and len(word) > 1]

def get_ngrams(words: List[str], n: int = 2, min_frequency: int = 1) -> Dict[str, int]:
    """Generate n-grams from a list of words with frequency filtering."""
    if len(words) < n:
        return {}
    
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

def calculate_tf_idf(term: str, document: str, all_documents: List[str]) -> float:
    """Calculates TF-IDF for a term in a document."""
    if not term or not document or not all_documents:
        return 0.0
    
    term_count = document.lower().split().count(term.lower())
    if term_count == 0:
        return 0.0
    
    doc_words = document.lower().split()
    if len(doc_words) == 0:
        return 0.0
    
    tf = term_count / len(doc_words)
    
    document_count = sum(1 for doc in all_documents if term.lower() in doc.lower())
    if document_count == 0:
        return 0.0
    
    idf = math.log(len(all_documents) / document_count)
    return tf * idf

def validate_url(url: str) -> bool:
    """Validate URL format."""
    return validators.url(url) is True

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def scrape_content(url: str, session: ClientSession, progress_callback: Optional[callable] = None) -> Optional[str]:
    """Scrapes content from a URL with retry logic and rate limiting."""
    try:
        if not validate_url(url):
            logger.error(f"Invalid URL format: {url}")
            return None
        
        async with RATE_LIMITER:
            if progress_callback:
                progress_callback(f"Scraping content from {url}")
            
            timeout = ClientTimeout(total=30)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with session.get(url, timeout=timeout, headers=headers) as response:
                response.raise_for_status()
                html = await response.text()
                
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove script and style tags
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.extract()
                
                # Get text content
                text = soup.get_text(separator=' ', strip=True)
                
                # Clean up text
                text = re.sub(r'\s+', ' ', text)
                text = text.strip()
                
                logger.info(f"Successfully scraped {len(text)} characters from {url}")
                return text
                
    except ClientError as e:
        logger.error(f"Client error scraping {url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error scraping {url}: {e}")
        raise

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_google_credentials_path():
    """Get Google Cloud credentials path from Streamlit secrets."""
    try:
        # For Streamlit Cloud, use secrets
        if 'google_cloud_credentials' in st.secrets:
            import tempfile
            import json
            
            # Create temporary file with credentials
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(dict(st.secrets['google_cloud_credentials']), f)
                return f.name
        else:
            # For local development
            return "./service_account.json"
    except Exception as e:
        logger.error(f"Failed to get Google credentials: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def analyze_content(content: str, all_documents: List[str], progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """Analyzes content using Google Cloud Natural Language API with enhanced error handling."""
    try:
        if not content or not content.strip():
            logger.warning("Empty content provided for analysis")
            return {"entities": {}, "document_sentiment": {"score": 0, "magnitude": 0}, "keyword_analysis": {}}
        
        if progress_callback:
            progress_callback("Analyzing content with Google Cloud Natural Language API")
        
        credentials_path = get_google_credentials_path()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        async with language_v1.LanguageServiceAsyncClient() as client:
            type_ = language_v1.Document.Type.PLAIN_TEXT
            
            # Truncate content if too long (API limit is 1MB)
            if len(content.encode('utf-8')) > 1000000:
                content = content[:500000]  # Truncate to ~500KB
                logger.warning("Content truncated due to API limits")
            
            document = {"content": content, "type_": type_}
            encoding_type = language_v1.EncodingType.UTF8

            try:
                # Analyze Entities
                if progress_callback:
                    progress_callback("Extracting entities")
                
                response = await client.analyze_entities(
                    request={"document": document, "encoding_type": encoding_type}
                )

                # Analyze Sentiment
                if progress_callback:
                    progress_callback("Analyzing sentiment")
                
                sentiment_response = await client.analyze_sentiment(
                    request={"document": document, "encoding_type": encoding_type}
                )

            except google_exceptions.GoogleAPIError as e:
                logger.error(f"Google API error: {e}")
                # Return empty results on API failure
                return {"entities": {}, "document_sentiment": {"score": 0, "magnitude": 0}, "keyword_analysis": {}}

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
            if progress_callback:
                progress_callback("Performing keyword analysis")
            
            words = tokenize(content)
            words = remove_stopwords(words)
            word_counts = Counter(words)
            total_words = len(words)

            # Phrase Extraction
            phrases = get_ngrams(words)
            phrase_counts = Counter(phrases)

            keyword_analysis = {}
            for entity_name, entity_data in entities.items():
                entity_words = tokenize(entity_name)
                entity_count = sum(word_counts.get(word, 0) for word in entity_words)
                
                keyword_analysis[entity_name] = {
                    "density": (entity_count / total_words) * 100 if total_words > 0 else 0,
                    "count": entity_count,
                    "phrase_counts": phrase_counts.get(entity_name, 0),
                    "tf_idf": calculate_tf_idf(entity_name, content, all_documents)
                }

            return {
                "entities": entities,
                "document_sentiment": {
                    "score": sentiment_response.document_sentiment.score,
                    "magnitude": sentiment_response.document_sentiment.magnitude
                },
                "keyword_analysis": keyword_analysis,
            }

    except Exception as e:
        logger.error(f"Error in analyze_content: {e}")
        # Return empty results on error
        return {"entities": {}, "document_sentiment": {"score": 0, "magnitude": 0}, "keyword_analysis": {}}

def compare_pages(client_analysis: Dict[str, Any], competitive_analyses: List[Dict[str, Any]], competitive_pages: List[str]) -> Dict[str, Any]:
    """Compares the client page analysis to the competitive pages with enhanced filtering."""
    try:
        client_entities = client_analysis.get("entities", {})
        client_keywords = client_analysis.get("keyword_analysis", {})

        missing_entities = {}
        missing_keywords = {}

        for i, comp_analysis in enumerate(competitive_analyses):
            if i >= len(competitive_pages):
                continue
                
            comp_entities = comp_analysis.get("entities", {})
            comp_keywords = comp_analysis.get("keyword_analysis", {})
            comp_url = competitive_pages[i]

            # Find missing entities
            for entity_name, entity_data in comp_entities.items():
                if entity_name not in client_entities and entity_data.get("salience", 0) > 0.01:  # Filter low salience
                    if entity_name not in missing_entities:
                        missing_entities[entity_name] = {
                            "competitors": {comp_url: {"salience": entity_data["salience"]}},
                            "type": entity_data["type"]
                        }
                    else:
                        missing_entities[entity_name]["competitors"][comp_url] = {"salience": entity_data["salience"]}

            # Find missing keywords
            for keyword, data in comp_keywords.items():
                if keyword not in client_keywords and data.get("density", 0) > 0.1:  # Filter low density
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
    
    except Exception as e:
        logger.error(f"Error in compare_pages: {e}")
        return {"missing_entities": {}, "missing_keywords": {}}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate_entity_recommendations(entity_item: EntitySelection, client_content: str, progress_callback: Optional[callable] = None) -> EntityRecommendations:
    """Generates structured recommendations for integrating a target entity."""
    try:
        if progress_callback:
            progress_callback(f"Generating recommendations for {entity_item.entity_name}")
        
        instructions = """
        You are an expert SEO content strategist. Your task is to analyze a client's webpage content and provide specific, actionable recommendations on how to integrate a target entity effectively.

        **Context:**

        * **Entities are more than just keywords:** They represent topics, concepts, or ideas. Effective integration requires understanding the entity's meaning, related terms, and user intent.
        * **Avoid keyword stuffing:** Do not simply repeat the entity's name throughout the content. Focus on natural language and readability.
        * **Strategic placement is key:** Consider where the entity and related terms will fit naturally within the content (e.g., title, headings, introduction, body, conclusion).
        * **Provide value to users:** Ensure the content is informative, engaging, and addresses user needs.
        * **Use related terms:** Incorporate synonyms, LSI keywords, and related concepts to expand the semantic scope.

        **Input:**

        * **Target Entity Info:** 
        * **Entity Name**: "{entity_name}"
        * **Relevance Score**: {relevance_score}
        * **Reasoning**: {reasoning}
        * **Client Page Content:**
        ```
        {client_page_content}
        ```

        **Instructions:**

        1. **Analyze the Client Page Content:** Review the provided content to understand its current focus, structure, and tone.
        2. **Research the Target Entity:** Understand the meaning of "{entity_name}", its key aspects, related terms, and user intent.
        3. **Identify Integration Opportunities:** Determine where the entity and related terms can be naturally incorporated into the existing content.
        4. **Provide Specific Recommendations:**
           * Suggest specific sections or paragraphs where the entity can be discussed.
           * Recommend related terms and concepts to use alongside the entity.
           * Provide examples of how to phrase sentences and paragraphs to incorporate the entity naturally.
           * Suggest where to place the entity in key areas like the title tag, meta description, and headings.
           * Explain *why* each recommendation is beneficial for both SEO and user experience.
        5. **Focus on User Value:** Ensure that the recommendations will result in content that is valuable, informative, and engaging for users.
        6. **Avoid Keyword Stuffing:** Do not recommend simply repeating the entity's name throughout the content.
        """
        
        # Truncate content if too long for API
        truncated_content = client_content[:8000] if len(client_content) > 8000 else client_content
        
        prompt = instructions.format(
            entity_name=entity_item.entity_name, 
            relevance_score=entity_item.relevance_score, 
            reasoning=entity_item.reasoning, 
            client_page_content=truncated_content
        )
        
        inputs_for_recommendation = [
            ("system", "Provide a structured recommendation for integrating the target entity."), 
            ("user", prompt)
        ]
        
        structured_recommendation_model = model.with_structured_output(EntityRecommendations)
        output = await structured_recommendation_model.ainvoke(inputs_for_recommendation)
        
        logger.info(f'EntityRecommendation generated for: {entity_item.entity_name}')
        return output
        
    except Exception as e:
        logger.error(f"Error generating recommendations for {entity_item.entity_name}: {e}")
        # Return a basic recommendation structure on error
        return EntityRecommendations(
            entity_context=entity_item.__dict__,
            integration_opportunities=[
                IntegrationOpportunity(
                    section="Content Enhancement",
                    recommendation=f"Consider incorporating {entity_item.entity_name} into relevant sections of your content.",
                    related_terms=[entity_item.entity_name],
                    examples=[f"Add mentions of {entity_item.entity_name} where contextually appropriate."],
                    placement="Throughout the content where relevant",
                    explanation="This entity was identified as relevant based on competitor analysis."
                )
            ]
        )

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def select_entities_for_integration(missing_entities: Dict[str, Any], progress_callback: Optional[callable] = None) -> EntitySelections:
    """Selects the most relevant entities for integration using AI."""
    try:
        if not missing_entities:
            logger.warning("No missing entities provided for selection")
            return EntitySelections(selected_entities=[])

        if progress_callback:
            progress_callback("Selecting most relevant entities for integration")

        # Get array of objects with these keys
        entities = []
        for entity_name, data in missing_entities.items():
            competitors_data = data.get("competitors", {})
            if competitors_data:
                entities.append({
                    "entity_name": entity_name,
                    "entity_type": data.get("type", "UNKNOWN"),
                    "count_of_competitors_with_entity": len(competitors_data),
                    "max_salience": max([comp.get("salience", 0) for comp in competitors_data.values()]),
                    "competitors": list(competitors_data.keys())
                })

        if not entities:
            logger.warning("No valid entities found for selection")
            return EntitySelections(selected_entities=[])

        # Sort by relevance (combination of competitor count and max salience)
        entities.sort(key=lambda x: (x["count_of_competitors_with_entity"], x["max_salience"]), reverse=True)
        
        # Take top 15 entities for analysis (to select top 10)
        entities = entities[:15]

        structured_string_of_entities = "\n".join([
            f"- Entity Name: {entity['entity_name']}, Entity Type: {entity['entity_type']}, "
            f"Count of Competitors with Entity: {entity['count_of_competitors_with_entity']}, "
            f"Max Salience: {entity['max_salience']:.3f}, Competitors: {entity['competitors']}"
            for entity in entities
        ])

        prompt = """
        You are an expert SEO content strategist. Your task is to analyze a list of missing entities and select the top 10 most relevant entities to integrate into a client's webpage.

        **Context:**

        * **Entities are more than just keywords:** They represent topics, concepts, or ideas. Effective integration requires understanding the entity's meaning, related terms, and user intent.
        * **Relevance is key:** Select entities that are most relevant to the client's page content and user intent.
        * **Prioritize entities:** Prioritize entities that are most likely to improve the client's page ranking and user experience.
        * **Provide a relevance score:** Provide a relevance score between 0 and 1 for each entity.
        * **Provide reasoning:** Provide reasoning behind the selection of each entity.

        **Input:**

        * **Missing Entities (includes the entity name, count of competitors with the entity, and maximum salience. use this context to determine relevance):**
        ```
        {entity_details}
        ```

        **Instructions:**

        1. **Analyze the Missing Entities:** Review the provided list of missing entities to understand their meaning and relevance.
        2. **Select Top 10 Entities:** Select the top 10 most relevant entities to integrate into the client's webpage.
        3. **Provide a List of Entities:** Provide a list of the selected entities with their relevance scores and reasoning.
        """

        inputs_for_selection = [
            ("system", "Select the top 10 most relevant entities to integrate with relevance scores and reasoning."), 
            ("user", prompt.format(entity_details=structured_string_of_entities))
        ]

        structured_selection_model = model.with_structured_output(EntitySelections)
        output = await structured_selection_model.ainvoke(inputs_for_selection)

        logger.info(f'Selected {len(output.selected_entities)} entities for integration')
        return output

    except Exception as e:
        logger.error(f"Error selecting entities for integration: {e}")
        return EntitySelections(selected_entities=[])

# Progress tracking utility
class ProgressTracker:
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.progress_bar = None
        self.status_text = None
    
    def initialize_ui(self):
        """Initialize Streamlit UI components for progress tracking."""
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
    
    def update(self, description: str):
        """Update progress."""
        self.current_step += 1
        progress = min(self.current_step / self.total_steps, 1.0)
        
        if self.progress_bar:
            self.progress_bar.progress(progress)
        if self.status_text:
            self.status_text.text(f"Step {self.current_step}/{self.total_steps}: {description}")
        
        logger.info(f"Progress: {self.current_step}/{self.total_steps} - {description}")
    
    def complete(self):
        """Mark progress as complete."""
        if self.progress_bar:
            self.progress_bar.progress(1.0)
        if self.status_text:
            self.status_text.text("Analysis complete!")
