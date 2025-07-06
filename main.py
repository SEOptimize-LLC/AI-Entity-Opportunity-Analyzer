import streamlit as st
import asyncio
import aiohttp
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import json
import tempfile
import os

# Custom imports
from utils import (
    get_competitor_name, validate_urls, create_excel_report,
    display_analysis_summary, display_entity_visualizations,
    export_to_json, create_pdf_report
)
from data_models import FinalState, AnalysisMetadata
from entity_analysis import (
    analyze_content, scrape_content, compare_pages,
    select_entities_for_integration, generate_entity_recommendations,
    ProgressTracker
)

# Page configuration
st.set_page_config(
    page_title="AI Entity Opportunity Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #444;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitApp:
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        if 'current_analysis_id' not in st.session_state:
            st.session_state.current_analysis_id = None
    
    def validate_api_keys(self) -> bool:
        """Validate required API keys."""
        required_keys = ['OPENAI_API_KEY']
        missing_keys = []
        
        for key in required_keys:
            if key not in st.secrets and key not in os.environ:
                missing_keys.append(key)
        
        if missing_keys:
            st.error(f"Missing API keys: {', '.join(missing_keys)}")
            st.info("Please add the required API keys to your Streamlit secrets or environment variables.")
            return False
        return True
    
    def render_sidebar(self):
        """Render the sidebar with navigation and settings."""
        with st.sidebar:
            st.image("https://via.placeholder.com/200x100/1f77b4/ffffff?text=AI+Entity+Analyzer", width=200)
            
            st.markdown("---")
            
            # Navigation menu
            selected = option_menu(
                menu_title="Navigation",
                options=["🏠 Home", "📊 Analysis", "📈 Results", "⚙️ Settings"],
                icons=["house", "graph-up", "bar-chart", "gear"],
                menu_icon="cast",
                default_index=0,
                orientation="vertical"
            )
            
            st.markdown("---")
            
            # Analysis history
            if st.session_state.analysis_history:
                st.subheader("📋 Analysis History")
                for i, analysis in enumerate(st.session_state.analysis_history[-5:]):  # Show last 5
                    with st.expander(f"Analysis {i+1} - {analysis['timestamp'][:10]}"):
                        st.text(f"Client: {analysis['client_url'][:50]}...")
                        st.text(f"Competitors: {len(analysis['competitor_urls'])}")
                        if st.button(f"Load Analysis {i+1}", key=f"load_{i}"):
                            st.session_state.analysis_results = analysis
                            st.rerun()
        
        return selected
    
    def render_home_page(self):
        """Render the home page."""
        st.markdown('<div class="main-header">🔍 AI Entity Opportunity Analyzer</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ## Welcome to the AI Entity Opportunity Analyzer!
        
        This powerful tool helps you identify SEO opportunities by analyzing your website content against competitors using advanced AI and Natural Language Processing.
        
        ### 🚀 Key Features:
        - **Entity Analysis**: Discover entities and topics your competitors are covering
        - **AI-Powered Recommendations**: Get specific, actionable SEO recommendations
        - **Competitive Analysis**: Compare your content against multiple competitors
        - **Interactive Visualizations**: Understand your data through charts and graphs
        - **Multiple Export Formats**: Download results in Excel, JSON, or PDF formats
        
        ### 📈 How It Works:
        1. **Input URLs**: Provide your website URL and competitor URLs
        2. **Content Analysis**: Our AI analyzes content using Google Cloud Natural Language API
        3. **Entity Extraction**: Identifies entities, keywords, and topics
        4. **Competitive Comparison**: Finds missing opportunities
        5. **AI Recommendations**: Generates specific recommendations for improvement
        
        ### 🛠️ Technologies Used:
        - Google Cloud Natural Language API for entity recognition
        - OpenAI GPT-4 for intelligent recommendations
        - Advanced web scraping with rate limiting
        - Interactive visualizations with Plotly
        
        Ready to get started? Click on **Analysis** in the sidebar to begin!
        """)
        
        # Quick start section
        st.markdown("---")
        st.subheader("🎯 Quick Start")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Step 1: Prepare URLs**
            - Your website URL
            - 2-5 competitor URLs
            - Ensure URLs are accessible
            """)
        
        with col2:
            st.markdown("""
            **Step 2: Run Analysis**
            - Enter URLs in the Analysis tab
            - Click "Start Analysis"
            - Wait for processing (2-5 minutes)
            """)
        
        with col3:
            st.markdown("""
            **Step 3: Review Results**
            - View interactive visualizations
            - Read AI recommendations
            - Export results for implementation
            """)
    
    def render_analysis_page(self):
        """Render the analysis page."""
        st.markdown('<div class="main-header">📊 Content Analysis</div>', unsafe_allow_html=True)
        
        if not self.validate_api_keys():
            return
        
        st.markdown("Enter your website URL and competitor URLs to begin the analysis.")
        
        # URL input form
        with st.form("url_input_form"):
            st.subheader("🌐 URL Configuration")
            
            client_url = st.text_input(
                "Your Website URL",
                placeholder="https://your-website.com/page",
                help="Enter the URL of the page you want to analyze"
            )
            
            st.subheader("🏢 Competitor URLs")
            competitor_urls = []
            
            # Dynamic competitor URL inputs
            num_competitors = st.slider("Number of Competitors", 1, 10, 3)
            
            for i in range(num_competitors):
                url = st.text_input(
                    f"Competitor {i+1} URL",
                    key=f"competitor_{i}",
                    placeholder=f"https://competitor{i+1}.com/page"
                )
                if url:
                    competitor_urls.append(url)
            
            # Analysis options
            st.subheader("⚙️ Analysis Options")
            col1, col2 = st.columns(2)
            
            with col1:
                include_sentiment = st.checkbox("Include Sentiment Analysis", value=True)
                include_visualizations = st.checkbox("Generate Visualizations", value=True)
            
            with col2:
                max_entities = st.slider("Maximum Entities to Analyze", 5, 20, 10)
                detailed_analysis = st.checkbox("Detailed Analysis Mode", value=False)
            
            submit_button = st.form_submit_button("🚀 Start Analysis", use_container_width=True)
        
        if submit_button:
            if not client_url:
                st.error("Please enter your website URL")
                return
            
            if not competitor_urls:
                st.error("Please enter at least one competitor URL")
                return
            
            # Validate URLs
            all_urls = [client_url] + competitor_urls
            is_valid, errors = validate_urls(all_urls)
            
            if not is_valid:
                st.error("URL Validation Errors:")
                for error in errors:
                    st.error(f"• {error}")
                return
            
            # Start analysis
            st.success("URLs validated successfully! Starting analysis...")
            
            # Run analysis
            analysis_results = self.run_analysis(
                client_url=client_url,
                competitor_urls=competitor_urls,
                include_sentiment=include_sentiment,
                max_entities=max_entities,
                detailed_analysis=detailed_analysis
            )
            
            if analysis_results:
                st.session_state.analysis_results = analysis_results
                st.session_state.analysis_history.append(analysis_results)
                st.success("Analysis completed successfully!")
                
                # Show quick summary
                self.show_quick_summary(analysis_results)
                
                st.info("Go to the **Results** tab to view detailed analysis and recommendations.")
    
    def run_analysis(self, client_url: str, competitor_urls: List[str], 
                    include_sentiment: bool = True, max_entities: int = 10, 
                    detailed_analysis: bool = False) -> Optional[Dict[str, Any]]:
        """Run the main analysis asynchronously."""
        try:
            # Create analysis metadata
            analysis_id = str(uuid.uuid4())
            metadata = AnalysisMetadata(
                analysis_id=analysis_id,
                timestamp=datetime.now(),
                user_id=None,  # Could be implemented with authentication
                analysis_duration=None
            )
            
            # Calculate total steps for progress tracking
            total_steps = len(competitor_urls) + 1 + 4  # scraping + analysis + comparison + selection + recommendations
            progress_tracker = ProgressTracker(total_steps)
            progress_tracker.initialize_ui()
            
            # Run async analysis
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    self.async_analysis(
                        client_url=client_url,
                        competitor_urls=competitor_urls,
                        metadata=metadata,
                        progress_tracker=progress_tracker,
                        include_sentiment=include_sentiment,
                        max_entities=max_entities,
                        detailed_analysis=detailed_analysis
                    )
                )
                return result
            finally:
                loop.close()
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return None
    
    async def async_analysis(self, client_url: str, competitor_urls: List[str], 
                           metadata: AnalysisMetadata, progress_tracker: ProgressTracker,
                           include_sentiment: bool = True, max_entities: int = 10,
                           detailed_analysis: bool = False) -> Dict[str, Any]:
        """Perform the actual analysis asynchronously."""
        start_time = datetime.now()
        
        try:
            async with aiohttp.ClientSession() as session:
                # Scrape content
                progress_tracker.update("Scraping client website")
                client_content = await scrape_content(client_url, session, progress_tracker.update)
                
                if not client_content:
                    raise Exception("Failed to scrape client website")
                
                # Scrape competitor content
                competitive_contents = []
                valid_competitor_urls = []
                
                for i, url in enumerate(competitor_urls):
                    progress_tracker.update(f"Scraping competitor {i+1}")
                    content = await scrape_content(url, session, progress_tracker.update)
                    if content:
                        competitive_contents.append(content)
                        valid_competitor_urls.append(url)
                
                if not competitive_contents:
                    raise Exception("Failed to scrape any competitor websites")
                
                all_documents = [client_content] + competitive_contents
                
                # Analyze content
                progress_tracker.update("Analyzing client content")
                client_analysis = await analyze_content(client_content, all_documents, progress_tracker.update)
                
                progress_tracker.update("Analyzing competitor content")
                competitive_analyses = []
                for i, content in enumerate(competitive_contents):
                    analysis = await analyze_content(content, all_documents, progress_tracker.update)
                    competitive_analyses.append(analysis)
                
                # Compare results
                progress_tracker.update("Comparing content with competitors")
                comparison_results = compare_pages(client_analysis, competitive_analyses, valid_competitor_urls)
                
                # Select entities
                progress_tracker.update("Selecting entities for integration")
                selected_entities = await select_entities_for_integration(
                    comparison_results.get("missing_entities", {}),
                    progress_tracker.update
                )
                
                # Generate recommendations
                progress_tracker.update("Generating AI recommendations")
                recommendations = []
                for entity in selected_entities.selected_entities[:max_entities]:
                    rec = await generate_entity_recommendations(entity, client_content, progress_tracker.update)
                    recommendations.append(rec)
                
                # Complete analysis
                progress_tracker.complete()
                
                # Calculate duration
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                metadata.analysis_duration = duration
                
                # Create final state
                final_state = FinalState(
                    metadata=metadata,
                    client_url=client_url,
                    competitor_urls=valid_competitor_urls,
                    analysis_results=[client_analysis] + competitive_analyses,
                    comparison_results=comparison_results,
                    selected_entities=selected_entities,
                    recommendation_overview=recommendations
                )
                
                return {
                    "final_state": final_state,
                    "client_url": client_url,
                    "competitor_urls": valid_competitor_urls,
                    "analysis_results": [client_analysis] + competitive_analyses,
                    "comparison_results": comparison_results,
                    "selected_entities": selected_entities,
                    "recommendations": recommendations,
                    "metadata": metadata.dict(),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            raise
    
    def show_quick_summary(self, analysis_results: Dict[str, Any]):
        """Show a quick summary of analysis results."""
        st.subheader("🎯 Quick Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Competitors Analyzed",
                len(analysis_results.get("competitor_urls", []))
            )
        
        with col2:
            missing_entities = analysis_results.get("comparison_results", {}).get("missing_entities", {})
            st.metric(
                "Missing Entities",
                len(missing_entities)
            )
        
        with col3:
            selected_entities = analysis_results.get("selected_entities", {})
            selected_count = len(selected_entities.selected_entities) if hasattr(selected_entities, 'selected_entities') else 0
            st.metric(
                "Selected for Integration",
                selected_count
            )
        
        with col4:
            recommendations = analysis_results.get("recommendations", [])
            st.metric(
                "AI Recommendations",
                len(recommendations)
            )
    
    def render_results_page(self):
        """Render the results page."""
        st.markdown('<div class="main-header">📈 Analysis Results</div>', unsafe_allow_html=True)
        
        if not st.session_state.analysis_results:
            st.info("No analysis results available. Please run an analysis first.")
            return
        
        analysis_results = st.session_state.analysis_results
        
        # Display analysis summary
        display_analysis_summary(analysis_results)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "🔍 Entities", "🎯 Recommendations", "📈 Visualizations", "📁 Export"
        ])
        
        with tab1:
            self.render_overview_tab(analysis_results)
        
        with tab2:
            self.render_entities_tab(analysis_results)
        
        with tab3:
            self.render_recommendations_tab(analysis_results)
        
        with tab4:
            self.render_visualizations_tab(analysis_results)
        
        with tab5:
            self.render_export_tab(analysis_results)
    
    def render_overview_tab(self, analysis_results: Dict[str, Any]):
        """Render the overview tab."""
        st.subheader("📋 Analysis Overview")
        
        # Basic information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Client URL:**")
            st.code(analysis_results.get("client_url", "N/A"))
            
            st.markdown("**Analysis Date:**")
            timestamp = analysis_results.get("timestamp", "N/A")
            if timestamp != "N/A":
                st.code(timestamp[:19])  # Show date and time
        
        with col2:
            st.markdown("**Competitor URLs:**")
            for i, url in enumerate(analysis_results.get("competitor_urls", []), 1):
                st.code(f"{i}. {url}")
        
        # Analysis statistics
        st.markdown("---")
        st.subheader("📊 Analysis Statistics")
        
        metadata = analysis_results.get("metadata", {})
        duration = metadata.get("analysis_duration", 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Analysis Duration", f"{duration:.1f} seconds" if duration else "N/A")
        
        with col2:
            client_entities = analysis_results.get("analysis_results", [{}])[0].get("entities", {})
            st.metric("Client Entities Found", len(client_entities))
        
        with col3:
            missing_entities = analysis_results.get("comparison_results", {}).get("missing_entities", {})
            st.metric("Missing Opportunities", len(missing_entities))
        
        # Sentiment analysis
        if analysis_results.get("analysis_results"):
            st.markdown("---")
            st.subheader("😊 Sentiment Analysis")
            
            sentiment_data = []
            for i, analysis in enumerate(analysis_results["analysis_results"]):
                source = "Client" if i == 0 else f"Competitor {i}"
                sentiment = analysis.get("document_sentiment", {})
                sentiment_data.append({
                    "Source": source,
                    "Score": sentiment.get("score", 0),
                    "Magnitude": sentiment.get("magnitude", 0)
                })
            
            df_sentiment = pd.DataFrame(sentiment_data)
            st.dataframe(df_sentiment, use_container_width=True)
    
    def render_entities_tab(self, analysis_results: Dict[str, Any]):
        """Render the entities tab."""
        st.subheader("🔍 Entity Analysis")
        
        # Selected entities for integration
        selected_entities = analysis_results.get("selected_entities")
        if selected_entities and hasattr(selected_entities, 'selected_entities'):
            st.markdown("### 🎯 Selected Entities for Integration")
            
            for i, entity in enumerate(selected_entities.selected_entities, 1):
                with st.expander(f"{i}. {entity.entity_name} (Score: {entity.relevance_score:.2f})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Type:** {entity.entity_type}")
                        st.markdown(f"**Relevance Score:** {entity.relevance_score:.2f}")
                    
                    with col2:
                        st.markdown(f"**Found in Competitors:** {len(entity.competitors)}")
                        st.markdown(f"**Competitors:** {', '.join(entity.competitors)}")
                    
                    st.markdown("**Reasoning:**")
                    st.markdown(entity.reasoning)
        
        # Missing entities table
        st.markdown("---")
        st.markdown("### 📊 Missing Entities Summary")
        
        missing_entities = analysis_results.get("comparison_results", {}).get("missing_entities", {})
        if missing_entities:
            entity_data = []
            for entity_name, data in missing_entities.items():
                competitors = data.get("competitors", {})
                max_salience = max([comp.get("salience", 0) for comp in competitors.values()]) if competitors else 0
                entity_data.append({
                    "Entity": entity_name,
                    "Type": data.get("type", "Unknown"),
                    "Competitor Count": len(competitors),
                    "Max Salience": f"{max_salience:.3f}",
                    "Competitors": ", ".join(competitors.keys())
                })
            
            df_entities = pd.DataFrame(entity_data)
            st.dataframe(df_entities, use_container_width=True)
        else:
            st.info("No missing entities found.")
    
    def render_recommendations_tab(self, analysis_results: Dict[str, Any]):
        """Render the recommendations tab."""
        st.subheader("🎯 AI-Powered Recommendations")
        
        recommendations = analysis_results.get("recommendations", [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"### {i}. {rec.entity_context.entity_name}")
                
                # Entity context
                with st.expander("📋 Entity Context"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Type:** {rec.entity_context.entity_type}")
                        st.markdown(f"**Relevance:** {rec.entity_context.relevance:.2f}")
                    with col2:
                        st.markdown("**Reasoning:**")
                        st.markdown(rec.entity_context.reasoning)
                
                # Integration opportunities
                st.markdown("**Integration Opportunities:**")
                for j, opportunity in enumerate(rec.integration_opportunities, 1):
                    with st.expander(f"Opportunity {j}: {opportunity.section}"):
                        st.markdown(f"**Recommendation:** {opportunity.recommendation}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Related Terms:**")
                            for term in opportunity.related_terms:
                                st.markdown(f"• {term}")
                        
                        with col2:
                            st.markdown("**Examples:**")
                            for example in opportunity.examples:
                                st.markdown(f"• {example}")
                        
                        st.markdown(f"**Placement:** {opportunity.placement}")
                        st.markdown(f"**Explanation:** {opportunity.explanation}")
                
                st.markdown("---")
        else:
            st.info("No recommendations available.")
    
    def render_visualizations_tab(self, analysis_results: Dict[str, Any]):
        """Render the visualizations tab."""
        st.subheader("📈 Interactive Visualizations")
        
        # Display visualizations
        display_entity_visualizations(analysis_results)
        
        # Additional custom visualizations
        self.create_custom_visualizations(analysis_results)
    
    def create_custom_visualizations(self, analysis_results: Dict[str, Any]):
        """Create custom visualizations for the analysis."""
        
        # Entity relevance scores
        selected_entities = analysis_results.get("selected_entities")
        if selected_entities and hasattr(selected_entities, 'selected_entities'):
            st.subheader("🎯 Entity Relevance Scores")
            
            entity_names = [entity.entity_name for entity in selected_entities.selected_entities]
            relevance_scores = [entity.relevance_score for entity in selected_entities.selected_entities]
            
            fig = px.bar(
                x=entity_names,
                y=relevance_scores,
                title="Selected Entities by Relevance Score",
                labels={"x": "Entities", "y": "Relevance Score"},
                color=relevance_scores,
                color_continuous_scale="viridis"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Missing entities heatmap
        missing_entities = analysis_results.get("comparison_results", {}).get("missing_entities", {})
        if missing_entities:
            st.subheader("🔥 Missing Entities Heatmap")
            
            # Create heatmap data
            entities = list(missing_entities.keys())
            competitor_urls = analysis_results.get("competitor_urls", [])
            
            heatmap_data = []
            for entity in entities:
                row = []
                for url in competitor_urls:
                    competitors = missing_entities[entity].get("competitors", {})
                    if url in competitors:
                        row.append(competitors[url].get("salience", 0))
                    else:
                        row.append(0)
                heatmap_data.append(row)
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=[f"Competitor {i+1}" for i in range(len(competitor_urls))],
                y=entities,
                colorscale="blues",
                text=[[f"{val:.3f}" for val in row] for row in heatmap_data],
                texttemplate="%{text}",
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="Entity Salience Across Competitors",
                xaxis_title="Competitors",
                yaxis_title="Entities",
                height=max(400, len(entities) * 30)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_export_tab(self, analysis_results: Dict[str, Any]):
        """Render the export tab."""
        st.subheader("📁 Export Analysis Results")
        
        st.markdown("Choose your preferred format to export the analysis results:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Excel Report")
            st.markdown("Comprehensive Excel workbook with multiple sheets containing detailed analysis data.")
            
            if st.button("Generate Excel Report", key="excel_export"):
                try:
                    excel_data = create_excel_report(analysis_results)
                    st.download_button(
                        label="Download Excel Report",
                        data=excel_data,
                        file_name=f"entity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"Error generating Excel report: {e}")
        
        with col2:
            st.markdown("### 📄 JSON Data")
            st.markdown("Raw analysis data in JSON format for programmatic use.")
            
            if st.button("Generate JSON Export", key="json_export"):
                try:
                    json_data = json.dumps(analysis_results, indent=2, default=str)
                    st.download_button(
                        label="Download JSON Data",
                        data=json_data,
                        file_name=f"entity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Error generating JSON export: {e}")
        
        with col3:
            st.markdown("### 📋 Markdown Report")
            st.markdown("Human-readable markdown report with recommendations.")
            
            if st.button("Generate Markdown Report", key="markdown_export"):
                try:
                    final_state = analysis_results.get("final_state")
                    if final_state:
                        markdown_content = final_state.to_markdown
                        st.download_button(
                            label="Download Markdown Report",
                            data=markdown_content,
                            file_name=f"entity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                    else:
                        st.error("Final state not available for markdown export")
                except Exception as e:
                    st.error(f"Error generating markdown report: {e}")
        
        # Preview section
        st.markdown("---")
        st.subheader("👀 Preview")
        
        preview_type = st.selectbox("Select preview type:", ["Summary", "Entities", "Recommendations"])
        
        if preview_type == "Summary":
            self.show_summary_preview(analysis_results)
        elif preview_type == "Entities":
            self.show_entities_preview(analysis_results)
        elif preview_type == "Recommendations":
            self.show_recommendations_preview(analysis_results)
    
    def show_summary_preview(self, analysis_results: Dict[str, Any]):
        """Show summary preview."""
        st.markdown("### 📋 Summary Preview")
        
        summary_data = {
            "Analysis ID": analysis_results.get("metadata", {}).get("analysis_id", "N/A"),
            "Client URL": analysis_results.get("client_url", "N/A"),
            "Competitors": len(analysis_results.get("competitor_urls", [])),
            "Missing Entities": len(analysis_results.get("comparison_results", {}).get("missing_entities", {})),
            "Selected Entities": len(analysis_results.get("selected_entities", {}).selected_entities) if hasattr(analysis_results.get("selected_entities", {}), 'selected_entities') else 0,
            "Recommendations": len(analysis_results.get("recommendations", []))
        }
        
        df_summary = pd.DataFrame(list(summary_data.items()), columns=["Metric", "Value"])
        st.dataframe(df_summary, use_container_width=True)
    
    def show_entities_preview(self, analysis_results: Dict[str, Any]):
        """Show entities preview."""
        st.markdown("### 🔍 Entities Preview")
        
        selected_entities = analysis_results.get("selected_entities")
        if selected_entities and hasattr(selected_entities, 'selected_entities'):
            entity_data = []
            for entity in selected_entities.selected_entities:
                entity_data.append({
                    "Entity": entity.entity_name,
                    "Type": entity.entity_type,
                    "Relevance": f"{entity.relevance_score:.2f}",
                    "Competitors": len(entity.competitors)
                })
            
            df_entities = pd.DataFrame(entity_data)
            st.dataframe(df_entities, use_container_width=True)
        else:
            st.info("No entities selected for preview.")
    
    def show_recommendations_preview(self, analysis_results: Dict[str, Any]):
        """Show recommendations preview."""
        st.markdown("### 🎯 Recommendations Preview")
        
        recommendations = analysis_results.get("recommendations", [])
        if recommendations:
            rec
