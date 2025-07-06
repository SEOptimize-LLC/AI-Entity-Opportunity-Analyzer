import asyncio
import aiohttp
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import re
from collections import Counter
import math

import validators
from bs4 import BeautifulSoup
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from asyncio_throttle import Throttler
from loguru import logger
import streamlit as st

# Correct import - using original filename
from data_models import (
    EntityAnalysis, ContentAnalysis, OpportunityEntity, 
    IntegrationRecommendation, AnalysisReport
)

# Rate limiting: 1 request per 2 seconds to be respectful
RATE_LIMITER = Throttler(rate_limit=1, period=2)

# Common stop words to filter out
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
    'will', 'with', 'this', 'but', 'they', 'have', 'had', 'what', 'when',
    'where', 'who', 'which', 'why', 'how', 'all', 'any', 'both', 'each'
}

class AIEntityAnalyzer:
    """Main analyzer class that uses only OpenAI for all analysis."""
    
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def scrape_content(self, url: str) -> Optional[Dict[str, str]]:
        """Enhanced scraping with anti-detection measures and better error handling."""
        try:
            if not validators.url(url):
                logger.error(f"Invalid URL format: {url}")
                return None
            
            async with RATE_LIMITER:
                # Enhanced headers to appear more like a real browser
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Sec-Fetch-User': '?1',
                    'Cache-Control': 'max-age=0',
                    'Pragma': 'no-cache'
                }
                
                timeout = aiohttp.ClientTimeout(total=45)  # Increased timeout
                
                try:
                    # Use connector with better SSL handling
                    connector = aiohttp.TCPConnector(
                        limit=10,
                        limit_per_host=5,
                        ttl_dns_cache=300,
                        use_dns_cache=True,
                        enable_cleanup_closed=True
                    )
                    
                    # Create session with connector if not already created
                    if not hasattr(self.session, '_connector') or self.session.closed:
                        self.session = aiohttp.ClientSession(connector=connector)
                    
                    async with self.session.get(
                        url, 
                        headers=headers, 
                        timeout=timeout, 
                        allow_redirects=True,
                        max_redirects=10,
                        ssl=False  # Allow self-signed certificates
                    ) as response:
                        
                        # Enhanced status code handling
                        if response.status == 403:
                            logger.warning(f"Access forbidden (403) for {url} - website may be blocking scrapers")
                            return None
                        elif response.status == 404:
                            logger.warning(f"Page not found (404) for {url}")
                            return None
                        elif response.status == 429:
                            logger.warning(f"Rate limited (429) for {url} - too many requests")
                            return None
                        elif response.status == 503:
                            logger.warning(f"Service unavailable (503) for {url}")
                            return None
                        elif response.status != 200:
                            logger.warning(f"HTTP {response.status} for {url}")
                            return None
                        
                        # Check content type
                        content_type = response.headers.get('content-type', '').lower()
                        if 'text/html' not in content_type and 'application/xhtml' not in content_type:
                            logger.warning(f"Non-HTML content type for {url}: {content_type}")
                            return None
                        
                        # Read content with proper encoding handling
                        try:
                            html = await response.text(encoding='utf-8')
                        except UnicodeDecodeError:
                            # Fallback to different encodings
                            try:
                                html = await response.text(encoding='latin-1')
                            except:
                                html = await response.text(errors='ignore')
                        
                        # Check for anti-bot protection
                        html_lower = html.lower()
                        bot_detection_phrases = [
                            'cloudflare', 'captcha', 'bot detection', 'access denied',
                            'blocked', 'security check', 'are you human', 'verification',
                            'suspicious activity', 'automated traffic'
                        ]
                        
                        if any(phrase in html_lower for phrase in bot_detection_phrases):
                            logger.warning(f"Bot protection detected for {url}")
                            return None
                        
                        # Check for minimal content (likely error pages)
                        if len(html) < 500:
                            logger.warning(f"Response too short for {url}: {len(html)} characters")
                            return None
                        
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Extract title with multiple fallbacks
                        title = "No title found"
                        title_tag = soup.find('title')
                        if title_tag:
                            title = title_tag.get_text().strip()
                        else:
                            # Try meta title
                            meta_title = soup.find('meta', property='og:title')
                            if meta_title:
                                title = meta_title.get('content', '').strip()
                            else:
                                # Try h1 as last resort
                                h1_tag = soup.find('h1')
                                if h1_tag:
                                    title = h1_tag.get_text().strip()
                                else:
                                    title = f"Page from {urlparse(url).netloc}"
                        
                        # Remove unwanted elements more thoroughly
                        unwanted_tags = [
                            'script', 'style', 'nav', 'footer', 'header', 'aside', 
                            'noscript', 'iframe', 'object', 'embed', 'applet',
                            'link', 'meta', 'base', 'comment'
                        ]
                        
                        for tag_name in unwanted_tags:
                            for tag in soup.find_all(tag_name):
                                tag.decompose()
                        
                        # Remove comments
                        for comment in soup.find_all(string=lambda text: isinstance(text, str) and text.strip().startswith('<!--')):
                            comment.extract()
                        
                        # Try to extract main content more intelligently
                        main_content = None
                        
                        # Look for main content containers
                        main_selectors = [
                            'main', '[role="main"]', '.main-content', '#main-content',
                            '.content', '#content', '.post-content', '.entry-content',
                            'article', '.article', '.post', '.entry'
                        ]
                        
                        for selector in main_selectors:
                            main_container = soup.select_one(selector)
                            if main_container:
                                main_content = main_container.get_text(separator=' ', strip=True)
                                break
                        
                        # Fallback to body content if no main content found
                        if not main_content:
                            body = soup.find('body')
                            if body:
                                main_content = body.get_text(separator=' ', strip=True)
                            else:
                                main_content = soup.get_text(separator=' ', strip=True)
                        
                        # Clean up content
                        content = re.sub(r'\s+', ' ', main_content)
                        content = content.strip()
                        
                        # Enhanced content validation
                        word_count = len(content.split())
                        
                        if word_count < 100:
                            logger.warning(f"Content too short for meaningful analysis: {url} ({word_count} words)")
                            return None
                        
                        # Check for duplicate/spam content patterns
                        words = content.lower().split()
                        unique_words = set(words)
                        if len(unique_words) < len(words) * 0.3:  # Less than 30% unique words
                            logger.warning(f"Content appears to be repetitive/spam for {url}")
                            return None
                        
                        logger.info(f"Successfully scraped {word_count} words from {url}")
                        
                        return {
                            'url': url,
                            'title': title[:200],  # Limit title length
                            'content': content,
                            'word_count': word_count
                        }
                        
                except aiohttp.ClientConnectorError as e:
                    logger.error(f"Connection error for {url}: {str(e)}")
                    return None
                except aiohttp.ServerTimeoutError:
                    logger.error(f"Server timeout for {url}")
                    return None
                except asyncio.TimeoutError:
                    logger.error(f"Request timeout for {url}")
                    return None
                except aiohttp.ClientResponseError as e:
                    logger.error(f"HTTP response error for {url}: {e.status} - {e.message}")
                    return None
                except Exception as e:
                    logger.error(f"Unexpected error scraping {url}: {str(e)}")
                    return None
                        
        except Exception as e:
            logger.error(f"Error in scrape_content for {url}: {str(e)}")
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def analyze_content_with_ai(self, content_data: Dict[str, str]) -> Optional[ContentAnalysis]:
        """Analyze content using OpenAI to extract entities, sentiment, and topics."""
        try:
            # Truncate content if too long (to stay within API limits)
            content = content_data['content']
            if len(content) > 8000:
                content = content[:8000] + "..."
            
            prompt = f"""
            Analyze the following webpage content and extract:
            1. Named entities (people, organizations, locations, products, etc.)
            2. Overall sentiment score (-1 to 1, where -1 is very negative, 0 is neutral, 1 is very positive)
            3. Main topics covered
            4. For each entity, provide:
               - Entity name
               - Entity type (PERSON, ORGANIZATION, LOCATION, PRODUCT, EVENT, etc.)
               - Salience score (0-1, how important/prominent this entity is in the content)
               - Sentiment score (-1 to 1, sentiment toward this specific entity)
               - Number of mentions (estimate)
            
            Content to analyze:
            Title: {content_data['title']}
            Content: {content}
            
            Please provide a structured JSON response with this exact format:
            {{
                "entities": [
                    {{
                        "name": "Entity Name",
                        "type": "ENTITY_TYPE",
                        "salience": 0.8,
                        "sentiment": 0.1,
                        "mentions": 3
                    }}
                ],
                "overall_sentiment": 0.2,
                "topics": ["topic1", "topic2", "topic3"]
            }}
            
            Focus on entities that are relevant for SEO and content strategy.
            """
            
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert SEO content analyst. Provide accurate, structured analysis in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            # Parse the AI response
            ai_response = response.choices[0].message.content
            
            # Extract JSON from the response
            try:
                # Try to find JSON in the response
                json_start = ai_response.find('{')
                json_end = ai_response.rfind('}') + 1
                json_str = ai_response[json_start:json_end]
                analysis_data = json.loads(json_str)
            except:
                logger.error("Failed to parse AI response as JSON")
                return None
            
            # Convert to our data model
            entities = []
            for entity_data in analysis_data.get('entities', []):
                try:
                    entity = EntityAnalysis(
                        name=entity_data['name'],
                        type=entity_data['type'],
                        salience=float(entity_data.get('salience', 0)),
                        sentiment=float(entity_data.get('sentiment', 0)),
                        mentions=int(entity_data.get('mentions', 1))
                    )
                    entities.append(entity)
                except Exception as e:
                    logger.warning(f"Skipping invalid entity: {entity_data}")
                    continue
            
            return ContentAnalysis(
                url=content_data['url'],
                title=content_data['title'],
                word_count=content_data['word_count'],
                entities=entities,
                overall_sentiment=float(analysis_data.get('overall_sentiment', 0)),
                topics=analysis_data.get('topics', [])
            )
            
        except Exception as e:
            logger.error(f"Error analyzing content: {str(e)}")
            return None
    
    async def find_opportunities(self, client_analysis: ContentAnalysis, 
                               competitor_analyses: List[ContentAnalysis]) -> List[OpportunityEntity]:
        """Find SEO opportunities by comparing client content with competitors."""
        try:
            # Get client entities
            client_entities = {entity.name.lower(): entity for entity in client_analysis.entities}
            
            # Find entities that appear in competitors but not in client
            competitor_entities = {}
            
            for comp_analysis in competitor_analyses:
                for entity in comp_analysis.entities:
                    entity_name = entity.name.lower()
                    if entity_name not in client_entities:
                        if entity_name not in competitor_entities:
                            competitor_entities[entity_name] = {
                                'entity': entity,
                                'found_in': [comp_analysis.url],
                                'salience_scores': [entity.salience],
                                'types': [entity.type]
                            }
                        else:
                            competitor_entities[entity_name]['found_in'].append(comp_analysis.url)
                            competitor_entities[entity_name]['salience_scores'].append(entity.salience)
                            competitor_entities[entity_name]['types'].append(entity.type)
            
            # Convert to opportunities (entities found in multiple competitors)
            opportunities = []
            for entity_name, data in competitor_entities.items():
                if len(data['found_in']) >= 2:  # Found in at least 2 competitors
                    avg_salience = sum(data['salience_scores']) / len(data['salience_scores'])
                    most_common_type = Counter(data['types']).most_common(1)[0][0]
                    
                    # Calculate relevance score
                    relevance_score = min(1.0, avg_salience * (len(data['found_in']) / len(competitor_analyses)))
                    
                    opportunity = OpportunityEntity(
                        name=data['entity'].name,
                        type=most_common_type,
                        relevance_score=relevance_score,
                        found_in_competitors=data['found_in'],
                        avg_competitor_salience=avg_salience,
                        reasoning=f"Found in {len(data['found_in'])} competitors with average salience of {avg_salience:.2f}"
                    )
                    opportunities.append(opportunity)
            
            # Sort by relevance score
            opportunities.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return opportunities[:10]  # Return top 10 opportunities
            
        except Exception as e:
            logger.error(f"Error finding opportunities: {str(e)}")
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_recommendations(self, client_analysis: ContentAnalysis, 
                                     opportunities: List[OpportunityEntity]) -> List[IntegrationRecommendation]:
        """Generate specific recommendations for integrating opportunities."""
        try:
            recommendations = []
            
            for opportunity in opportunities[:5]:  # Top 5 opportunities
                prompt = f"""
                I need specific SEO recommendations for integrating an entity into webpage content.
                
                CLIENT PAGE CONTEXT:
                - URL: {client_analysis.url}
                - Title: {client_analysis.title}
                - Current topics: {', '.join(client_analysis.topics)}
                - Word count: {client_analysis.word_count}
                
                OPPORTUNITY ENTITY:
                - Name: {opportunity.name}
                - Type: {opportunity.type}
                - Relevance Score: {opportunity.relevance_score:.2f}
                - Found in {len(opportunity.found_in_competitors)} competitors
                - Reasoning: {opportunity.reasoning}
                
                Please provide specific, actionable recommendations:
                1. Where to integrate this entity (specific section)
                2. Exact recommendation for integration
                3. Example text showing how to incorporate naturally
                4. Expected SEO impact
                
                Provide response in JSON format:
                {{
                    "section": "Where to integrate (e.g., 'Introduction', 'Product Features', 'About Us')",
                    "recommendation": "Specific actionable recommendation",
                    "example_text": "Example sentence or paragraph showing integration",
                    "seo_impact": "Expected SEO benefit"
                }}
                
                Focus on natural integration that adds value to users while improving SEO.
                """
                
                response = await self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert SEO content strategist. Provide specific, actionable recommendations."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                
                try:
                    ai_response = response.choices[0].message.content
                    json_start = ai_response.find('{')
                    json_end = ai_response.rfind('}') + 1
                    json_str = ai_response[json_start:json_end]
                    rec_data = json.loads(json_str)
                    
                    recommendation = IntegrationRecommendation(
                        entity_name=opportunity.name,
                        section=rec_data['section'],
                        recommendation=rec_data['recommendation'],
                        example_text=rec_data['example_text'],
                        seo_impact=rec_data['seo_impact']
                    )
                    recommendations.append(recommendation)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse recommendation for {opportunity.name}: {e}")
                    continue
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    async def run_complete_analysis(self, client_url: str, competitor_urls: List[str], 
                                  progress_callback: Optional[callable] = None) -> Optional[AnalysisReport]:
        """Run complete SEO analysis and return structured report."""
        try:
            analysis_id = str(uuid.uuid4())
            
            if progress_callback:
                progress_callback("Starting content analysis...")
            
            # Scrape client content
            if progress_callback:
                progress_callback("Analyzing client webpage...")
            client_content = await self.scrape_content(client_url)
            if not client_content:
                raise Exception("Failed to scrape client content")
            
            # Analyze client content
            client_analysis = await self.analyze_content_with_ai(client_content)
            if not client_analysis:
                raise Exception("Failed to analyze client content")
            
            # Scrape and analyze competitors
            competitor_analyses = []
            successfully_scraped_urls = []
            for i, comp_url in enumerate(competitor_urls):
                if progress_callback:
                    progress_callback(f"Analyzing competitor {i+1}/{len(competitor_urls)}...")
                
                comp_content = await self.scrape_content(comp_url)
                if comp_content:
                    comp_analysis = await self.analyze_content_with_ai(comp_content)
                    if comp_analysis:
                        competitor_analyses.append(comp_analysis)
                        successfully_scraped_urls.append(comp_url)
            
            if not competitor_analyses:
                raise Exception("Failed to analyze any competitor content")
            
            # Find opportunities
            if progress_callback:
                progress_callback("Identifying SEO opportunities...")
            opportunities = await self.find_opportunities(client_analysis, competitor_analyses)
            
            # Generate recommendations
            if progress_callback:
                progress_callback("Generating AI recommendations...")
            recommendations = await self.generate_recommendations(client_analysis, opportunities)
            
            # Create final report
            report = AnalysisReport(
                analysis_id=analysis_id,
                timestamp=datetime.now(),
                client_url=client_url,
                competitor_urls=successfully_scraped_urls,  # Use only successful URLs
                client_analysis=client_analysis,
                competitor_analyses=competitor_analyses,
                missing_entities=opportunities,
                recommendations=recommendations,
                total_opportunities=len(opportunities),
                priority_opportunities=len([op for op in opportunities if op.relevance_score > 0.5])
            )
            
            if progress_callback:
                progress_callback("Analysis complete!")
            
            return report
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {str(e)}")
            if progress_callback:
                progress_callback(f"Error: {str(e)}")
            return None
