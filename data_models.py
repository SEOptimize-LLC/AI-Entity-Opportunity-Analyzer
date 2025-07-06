from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

class EntityAnalysis(BaseModel):
    """Analysis of a single entity found in content."""
    name: str = Field(description="Name of the entity")
    type: str = Field(description="Type of entity (PERSON, ORGANIZATION, etc.)")
    salience: float = Field(description="Importance score (0-1)")
    sentiment: float = Field(description="Sentiment score (-1 to 1)")
    mentions: int = Field(description="Number of times mentioned")
    
    @validator('salience')
    def validate_salience(cls, v):
        return max(0, min(1, v))
    
    @validator('sentiment')
    def validate_sentiment(cls, v):
        return max(-1, min(1, v))

class ContentAnalysis(BaseModel):
    """Complete analysis of a single webpage."""
    url: str = Field(description="URL of the analyzed page")
    title: str = Field(description="Page title")
    word_count: int = Field(description="Total word count")
    entities: List[EntityAnalysis] = Field(description="List of entities found")
    overall_sentiment: float = Field(description="Overall page sentiment")
    topics: List[str] = Field(description="Main topics identified")
    
class OpportunityEntity(BaseModel):
    """An entity that represents an SEO opportunity."""
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type")
    relevance_score: float = Field(description="How relevant this entity is (0-1)")
    found_in_competitors: List[str] = Field(description="URLs where this entity was found")
    avg_competitor_salience: float = Field(description="Average salience across competitors")
    reasoning: str = Field(description="Why this entity is an opportunity")

class IntegrationRecommendation(BaseModel):
    """Specific recommendation for integrating an entity."""
    entity_name: str = Field(description="Name of the entity to integrate")
    section: str = Field(description="Where to integrate (e.g., 'Main Content', 'Headers')")
    recommendation: str = Field(description="Specific recommendation")
    example_text: str = Field(description="Example of how to incorporate")
    seo_impact: str = Field(description="Expected SEO impact")

class AnalysisReport(BaseModel):
    """Complete analysis report with all findings."""
    analysis_id: str = Field(description="Unique identifier for this analysis")
    timestamp: datetime = Field(description="When analysis was performed")
    client_url: str = Field(description="Client webpage URL")
    competitor_urls: List[str] = Field(description="Competitor URLs analyzed")
    
    # Analysis results
    client_analysis: ContentAnalysis = Field(description="Client page analysis")
    competitor_analyses: List[ContentAnalysis] = Field(description="Competitor analyses")
    
    # Opportunities
    missing_entities: List[OpportunityEntity] = Field(description="Entities missing from client page")
    recommendations: List[IntegrationRecommendation] = Field(description="Specific recommendations")
    
    # Summary metrics
    total_opportunities: int = Field(description="Total number of opportunities found")
    priority_opportunities: int = Field(description="High-priority opportunities")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp.isoformat(),
            "client_url": self.client_url,
            "competitor_urls": self.competitor_urls,
            "client_analysis": self.client_analysis.dict(),
            "competitor_analyses": [comp.dict() for comp in self.competitor_analyses],
            "missing_entities": [ent.dict() for ent in self.missing_entities],
            "recommendations": [rec.dict() for rec in self.recommendations],
            "total_opportunities": self.total_opportunities,
            "priority_opportunities": self.priority_opportunities
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
