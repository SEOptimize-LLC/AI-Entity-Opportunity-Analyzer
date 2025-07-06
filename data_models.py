from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import streamlit as st
from utils import create_excel_report
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

### START: ENTITY SELECTION MODEL ###

class EntitySelection(BaseModel):
    """Entity name with reasoning behind the selection and score of how relevant it is for the client page."""
    entity_name: str = Field(name="entity_name", description="The name of the entity.")
    entity_type: str = Field(name="entity_type", description="The type of the entity.")
    relevance_score: float = Field(name="relevance_score", description="The relevance score of the entity for the client page (0-1).")
    reasoning: str = Field(name="reasoning", description="The reasoning behind the selection of the entity.")
    competitors: List[str] = Field(name="competitors", description="List of competitors where the entity was found.")
    
    @validator('relevance_score')
    def validate_relevance_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Relevance score must be between 0 and 1')
        return v

class EntitySelections(BaseModel):
    """List of entity selections."""
    selected_entities: List[EntitySelection] = Field(name="selected_entities", description="List of selected entities with relevance scores and reasoning.")
    
    @property
    def to_markdown(self) -> str:
        """Convert the entity selections to a markdown string."""
        markdown = ""
        for entity in self.selected_entities:
            markdown += f"- **{entity.entity_name}**\n"
            markdown += f"  - **Relevance Score:** {entity.relevance_score}\n"
            markdown += f"  - **Reasoning:** {entity.reasoning}\n"
            markdown += f"  - **Competitors:** {', '.join(entity.competitors)}\n\n"
        return markdown
    
    def create_relevance_chart(self) -> go.Figure:
        """Create a relevance score chart for visualization."""
        entities = [entity.entity_name for entity in self.selected_entities]
        scores = [entity.relevance_score for entity in self.selected_entities]
        
        fig = go.Figure(data=[
            go.Bar(x=entities, y=scores, marker_color='lightblue')
        ])
        
        fig.update_layout(
            title="Entity Relevance Scores",
            xaxis_title="Entities",
            yaxis_title="Relevance Score",
            yaxis=dict(range=[0, 1])
        )
        
        return fig

### END: ENTITY SELECTION MODEL ###

### START: ENTITY RECOMMENDATION MODEL ###

class CompetitorData(BaseModel):
    """Represents data from a competitor."""
    salience: Optional[float] = Field(name="salience", description="The salience score of the entity in the competitor's content.", default=None)
    density: Optional[float] = Field(name="density", description="The density of the keyword in the competitor's content.", default=None)
    count: Optional[int] = Field(name="count", description="The number of times the keyword appears in the competitor's content.", default=None)
    tf_idf: Optional[float] = Field(name="tf_idf", description="The TF-IDF score of the keyword in the competitor's content.", default=None)

class MissingItem(BaseModel):
    """Represents a missing item (entity or keyword)."""
    entity_name: str = Field(name="entity_name", description="The name of the missing item.")
    entity_type: str = Field(name="type", description="The exact type the user has provided for the missing item.")
    relevance: float = Field(name="relevance", description="The exact relevance score the user has provided for the missing item.")
    reasoning: str = Field(name="reasoning", description="The exact reasoning the user has provided for the missing item.")

class IntegrationOpportunity(BaseModel):
    """Represents an integration opportunity for an entity."""
    section: str = Field(name="section", description="The section or area of the content where the entity can be integrated.")
    recommendation: str = Field(name="recommendation", description="Specific recommendation on how to integrate the entity.")
    related_terms: List[str] = Field(name="related_terms", description="List of related terms to use alongside the entity.")
    examples: List[str] = Field(name="examples", description="Examples of how to incorporate the entity naturally.")
    placement: str = Field(name="placement", description="Suggested placement of the entity in key areas like the title tag, meta description, and headings.")
    explanation: str = Field(name="explanation", description="Explanation of why each recommendation is beneficial for SEO and user experience.")

class EntityRecommendations(BaseModel):
    """Structured recommendation for integrating an entity."""
    entity_context: MissingItem = Field(name="entity_context", description="Context about the entity and its competitors.")
    integration_opportunities: List[IntegrationOpportunity] = Field(name="integration_opportunities", description="List of integration opportunities for the entity.")
    
    @property
    def to_markdown(self) -> str:
        """Convert the entity recommendation to a markdown string."""
        markdown = f"### Entity Target: '{self.entity_context.entity_name.title()}'\n\n"
        for i, op in enumerate(self.integration_opportunities, start=1):
            markdown += f"#### Opportunity {i}: {op.section}\n\n"
            markdown += f"**Recommendation:** {op.recommendation}\n\n"
            markdown += f"**Related Terms:** {', '.join(op.related_terms)}\n\n"
            markdown += f"**Examples:**\n\n"
            for example in op.examples:
                markdown += f"- {example}\n"
            markdown += "\n"
            markdown += f"**Placement:** {op.placement}\n\n"
            markdown += f"**Explanation:** {op.explanation}\n\n"
        return markdown

### END: ENTITY RECOMMENDATION MODEL ###

### ANALYSIS METADATA ###

class AnalysisMetadata(BaseModel):
    """Metadata for the analysis session."""
    analysis_id: str = Field(description="Unique identifier for the analysis")
    timestamp: datetime = Field(description="When the analysis was performed")
    user_id: Optional[str] = Field(description="User who performed the analysis", default=None)
    analysis_duration: Optional[float] = Field(description="Duration of analysis in seconds", default=None)
    
class AnalysisProgress(BaseModel):
    """Track analysis progress for UI updates."""
    total_steps: int = Field(description="Total number of analysis steps")
    current_step: int = Field(description="Current step being processed")
    step_description: str = Field(description="Description of current step")
    progress_percentage: float = Field(description="Progress as percentage")
    
    @property
    def is_complete(self) -> bool:
        return self.current_step >= self.total_steps

### CONSOLIDATED DATA MODEL ###

class FinalState(BaseModel):
    """Represents the state of the analysis process."""
    metadata: AnalysisMetadata = Field(description="Analysis metadata")
    client_url: str = Field(name="client_url", description="The URL of the client's page.")
    competitor_urls: List[str] = Field(name="competitor_urls", description="List of competitor URLs.")
    analysis_results: List[dict] = Field(name="analysis_results", description="Results of the content analysis including the client and competitor entities and keywords.")
    comparison_results: dict = Field(name="comparison_results", description="Comparison results between the client and competitors.")
    selected_entities: EntitySelections = Field(name="selected_entities", description="List of selected entities with relevance scores and reasoning.")
    recommendation_overview: List[EntityRecommendations] = Field(name="recommendation_overview", description="Structured recommendation for integrating entities.")
    
    @property
    def to_markdown(self) -> str:
        """Create a markdown report from the analysis results."""
        markdown = f"# Content Analysis Report\n\n"
        markdown += f"**Analysis ID:** {self.metadata.analysis_id}\n"
        markdown += f"**Date:** {self.metadata.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        markdown += f"**Client Page:** {self.client_url}\n\n"
        markdown += "**Competitor Pages:**\n\n"
        for url in self.competitor_urls:
            markdown += f"- {url}\n"
        markdown += "\n"
        markdown += "\n"
        markdown += "## Selected Entities for Integration\n\n"
        markdown += self.selected_entities.to_markdown
        markdown += "\n"
        markdown += "## Entity Recommendations\n\n"
        for rec in self.recommendation_overview:
            markdown += rec.to_markdown
        return markdown
    
    @property
    def to_excel(self) -> bytes:
        """Create an Excel report from the analysis results."""
        return create_excel_report({
            "client_url": self.client_url,
            "competitor_urls": self.competitor_urls,
            "analysis_results": self.analysis_results,
            "comparison_results": self.comparison_results,
            "metadata": self.metadata.dict()
        })
    
    def create_summary_dashboard(self) -> Dict[str, Any]:
        """Create summary data for dashboard visualization."""
        return {
            "total_entities_analyzed": len(self.comparison_results.get("missing_entities", {})),
            "selected_entities_count": len(self.selected_entities.selected_entities),
            "competitor_count": len(self.competitor_urls),
            "avg_relevance_score": sum(e.relevance_score for e in self.selected_entities.selected_entities) / len(self.selected_entities.selected_entities) if self.selected_entities.selected_entities else 0,
            "analysis_duration": self.metadata.analysis_duration
        }
