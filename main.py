import streamlit as st
import asyncio
import json
import io
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
import validators

# Fixed import - using original filename
from analysis_engine import AIEntityAnalyzer
from data_models import AnalysisReport

# Page configuration
st.set_page_config(
    page_title="AI SEO Entity Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class SEOAnalyzerApp:
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state variables."""
        if 'analysis_report' not in st.session_state:
            st.session_state.analysis_report = None
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
    
    def validate_api_key(self) -> bool:
        """Validate OpenAI API key."""
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
            if not api_key:
                st.error("❌ OpenAI API key not found in secrets")
                st.info("Please add your OpenAI API key to the secrets.toml file")
                return False
            return True
        except Exception as e:
            st.error(f"❌ Error accessing API key: {str(e)}")
            return False
    
    def render_header(self):
        """Render the main header."""
        st.markdown('<div class="main-header">🔍 AI SEO Entity Analyzer</div>', unsafe_allow_html=True)
        st.markdown("**Discover SEO opportunities by analyzing competitor content with AI**")
        st.markdown("---")
    
    def render_sidebar(self):
        """Render the sidebar with navigation."""
        with st.sidebar:
            st.markdown("## 🎯 Navigation")
            
            # Analysis history
            if st.session_state.analysis_history:
                st.markdown("### 📋 Recent Analyses")
                for i, analysis in enumerate(st.session_state.analysis_history[-3:]):
                    if st.button(f"📊 Analysis {i+1}", key=f"load_{i}"):
                        st.session_state.analysis_report = analysis
                        st.rerun()
            
            st.markdown("---")
            
            # Quick stats
            if st.session_state.analysis_report:
                report = st.session_state.analysis_report
                st.markdown("### 📈 Current Analysis")
                st.metric("Opportunities Found", report.total_opportunities)
                st.metric("Priority Items", report.priority_opportunities)
                st.metric("Competitors", len(report.competitor_analyses))
            
            st.markdown("---")
            st.markdown("### ℹ️ About")
            st.info(
                "This tool uses AI to analyze your content against competitors "
                "and find SEO opportunities. Simply enter your URL and competitor URLs to get started."
            )
    
    def render_input_form(self):
        """Render the main input form."""
        st.markdown("## 🌐 Enter URLs for Analysis")
        
        with st.form("analysis_form"):
            # Client URL
            client_url = st.text_input(
                "Your Website URL",
                placeholder="https://your-website.com",
                help="Enter the URL of the page you want to analyze"
            )
            
            # Competitor URLs
            st.markdown("### 🏢 Competitor URLs")
            num_competitors = st.slider("Number of Competitors", 1, 5, 3)
            
            competitor_urls = []
            for i in range(num_competitors):
                url = st.text_input(
                    f"Competitor {i+1} URL",
                    key=f"comp_{i}",
                    placeholder=f"https://competitor{i+1}.com"
                )
                if url.strip():
                    competitor_urls.append(url.strip())
            
            # Submit button
            submitted = st.form_submit_button("🚀 Start Analysis", use_container_width=True)
            
            if submitted:
                if not client_url:
                    st.error("Please enter your website URL")
                    return
                
                if not competitor_urls:
                    st.error("Please enter at least one competitor URL")
                    return
                
                # Validate URLs
                all_urls = [client_url] + competitor_urls
                invalid_urls = [url for url in all_urls if not validators.url(url)]
                
                if invalid_urls:
                    st.error(f"Invalid URLs: {', '.join(invalid_urls)}")
                    return
                
                # Run analysis
                self.run_analysis(client_url, competitor_urls)
    
    def run_analysis(self, client_url: str, competitor_urls: list):
        """Run the analysis with progress tracking."""
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(message: str):
            status_text.text(message)
        
        try:
            # Get API key
            api_key = st.secrets.get("OPENAI_API_KEY")
            if not api_key:
                st.error("OpenAI API key not found")
                return
            
            # Run analysis
            async def run_async_analysis():
                async with AIEntityAnalyzer(api_key) as analyzer:
                    return await analyzer.run_complete_analysis(
                        client_url, competitor_urls, update_progress
                    )
            
            # Execute async analysis
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                report = loop.run_until_complete(run_async_analysis())
            finally:
                loop.close()
            
            if report:
                st.session_state.analysis_report = report
                st.session_state.analysis_history.append(report)
                progress_bar.progress(100)
                status_text.success("✅ Analysis completed successfully!")
                st.rerun()
            else:
                st.error("❌ Analysis failed. Please check your URLs and try again.")
                
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()
    
    def render_results(self):
        """Render analysis results."""
        if not st.session_state.analysis_report:
            return
        
        report = st.session_state.analysis_report
        
        st.markdown("## 📊 Analysis Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Opportunities", report.total_opportunities)
        with col2:
            st.metric("Priority Items", report.priority_opportunities)
        with col3:
            st.metric("Competitors Analyzed", len(report.competitor_analyses))
        with col4:
            st.metric("Client Word Count", report.client_analysis.word_count)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Opportunities", "📋 Recommendations", "📈 Visualizations", "📁 Export"])
        
        with tab1:
            self.render_opportunities_tab(report)
        
        with tab2:
            self.render_recommendations_tab(report)
        
        with tab3:
            self.render_visualizations_tab(report)
        
        with tab4:
            self.render_export_tab(report)
    
    def render_opportunities_tab(self, report: AnalysisReport):
        """Render opportunities tab."""
        st.markdown("### 🎯 SEO Opportunities Found")
        
        if not report.missing_entities:
            st.info("No opportunities found. Your content might already be well-optimized!")
            return
        
        for i, opportunity in enumerate(report.missing_entities, 1):
            with st.expander(f"{i}. {opportunity.name} (Score: {opportunity.relevance_score:.2f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Entity Type:** {opportunity.type}")
                    st.markdown(f"**Relevance Score:** {opportunity.relevance_score:.2f}")
                    st.markdown(f"**Avg. Competitor Salience:** {opportunity.avg_competitor_salience:.2f}")
                
                with col2:
                    st.markdown(f"**Found in {len(opportunity.found_in_competitors)} competitors:**")
                    for comp_url in opportunity.found_in_competitors:
                        st.markdown(f"• {comp_url}")
                
                st.markdown(f"**Why this is an opportunity:** {opportunity.reasoning}")
    
    def render_recommendations_tab(self, report: AnalysisReport):
        """Render recommendations tab."""
        st.markdown("### 🚀 AI-Powered Recommendations")
        
        if not report.recommendations:
            st.info("No specific recommendations available.")
            return
        
        for i, rec in enumerate(report.recommendations, 1):
            st.markdown(f"### {i}. {rec.entity_name}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Target Section:** {rec.section}")
                st.markdown(f"**Recommendation:** {rec.recommendation}")
            
            with col2:
                st.markdown(f"**Expected SEO Impact:** {rec.seo_impact}")
            
            st.markdown("**Example Integration:**")
            st.code(rec.example_text, language="text")
            
            st.markdown("---")
    
    def render_visualizations_tab(self, report: AnalysisReport):
        """Render visualizations tab."""
        st.markdown("### 📈 Data Visualizations")
        
        # Opportunity relevance chart
        if report.missing_entities:
            st.subheader("Opportunity Relevance Scores")
            
            entity_names = [entity.name for entity in report.missing_entities]
            relevance_scores = [entity.relevance_score for entity in report.missing_entities]
            
            fig = px.bar(
                x=entity_names,
                y=relevance_scores,
                title="SEO Opportunities by Relevance Score",
                labels={"x": "Entities", "y": "Relevance Score"},
                color=relevance_scores,
                color_continuous_scale="viridis"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment comparison
        st.subheader("Sentiment Analysis Comparison")
        
        sources = ["Your Website"] + [f"Competitor {i+1}" for i in range(len(report.competitor_analyses))]
        sentiments = [report.client_analysis.overall_sentiment] + [comp.overall_sentiment for comp in report.competitor_analyses]
        
        fig = px.bar(
            x=sources,
            y=sentiments,
            title="Content Sentiment Comparison",
            labels={"x": "Source", "y": "Sentiment Score"},
            color=sentiments,
            color_continuous_scale="RdYlBu"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Word count comparison
        st.subheader("Content Length Comparison")
        
        word_counts = [report.client_analysis.word_count] + [comp.word_count for comp in report.competitor_analyses]
        
        fig = px.bar(
            x=sources,
            y=word_counts,
            title="Word Count Comparison",
            labels={"x": "Source", "y": "Word Count"},
            color=word_counts,
            color_continuous_scale="blues"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_export_tab(self, report: AnalysisReport):
        """Render export options tab."""
        st.markdown("### 📁 Export Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📊 Excel Report")
            if st.button("Generate Excel Report"):
                excel_data = self.create_excel_report(report)
                st.download_button(
                    label="Download Excel Report",
                    data=excel_data,
                    file_name=f"seo_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col2:
            st.markdown("#### 📄 JSON Data")
            if st.button("Generate JSON Export"):
                json_data = report.to_json()
                st.download_button(
                    label="Download JSON Data",
                    data=json_data,
                    file_name=f"seo_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col3:
            st.markdown("#### 📝 Summary Report")
            if st.button("Generate Summary"):
                summary = self.create_summary_report(report)
                st.download_button(
                    label="Download Summary",
                    data=summary,
                    file_name=f"seo_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    def create_excel_report(self, report: AnalysisReport) -> bytes:
        """Create Excel report from analysis results."""
        output = io.BytesIO()
        workbook = Workbook()
        
        # Remove default sheet
        workbook.remove(workbook.active)
        
        # Header styles
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        
        # Summary sheet
        summary_sheet = workbook.create_sheet("Summary")
        summary_data = [
            ["Metric", "Value"],
            ["Analysis Date", report.timestamp.strftime("%Y-%m-%d %H:%M:%S")],
            ["Client URL", report.client_url],
            ["Total Opportunities", report.total_opportunities],
            ["Priority Opportunities", report.priority_opportunities],
            ["Competitors Analyzed", len(report.competitor_analyses)]
        ]
        
        for row in summary_data:
            summary_sheet.append(row)
        
        # Style headers
        for cell in summary_sheet[1]:
            cell.font = header_font
            cell.fill = header_fill
        
        # Opportunities sheet
        opportunities_sheet = workbook.create_sheet("Opportunities")
        opp_headers = ["Entity", "Type", "Relevance Score", "Avg Salience", "Found in URLs", "Reasoning"]
        opportunities_sheet.append(opp_headers)
        
        for cell in opportunities_sheet[1]:
            cell.font = header_font
            cell.fill = header_fill
        
        for opp in report.missing_entities:
            opportunities_sheet.append([
                opp.name,
                opp.type,
                opp.relevance_score,
                opp.avg_competitor_salience,
                "; ".join(opp.found_in_competitors),
                opp.reasoning
            ])
        
        # Recommendations sheet
        rec_sheet = workbook.create_sheet("Recommendations")
        rec_headers = ["Entity", "Section", "Recommendation", "Example", "SEO Impact"]
        rec_sheet.append(rec_headers)
        
        for cell in rec_sheet[1]:
            cell.font = header_font
            cell.fill = header_fill
        
        for rec in report.recommendations:
            rec_sheet.append([
                rec.entity_name,
                rec.section,
                rec.recommendation,
                rec.example_text,
                rec.seo_impact
            ])
        
        workbook.save(output)
        output.seek(0)
        return output.getvalue()
    
    def create_summary_report(self, report: AnalysisReport) -> str:
        """Create a text summary report."""
        summary = f"""
SEO ENTITY ANALYSIS REPORT
=========================

Analysis Date: {report.timestamp.strftime("%Y-%m-%d %H:%M:%S")}
Client URL: {report.client_url}

SUMMARY METRICS
--------------
Total Opportunities Found: {report.total_opportunities}
Priority Opportunities: {report.priority_opportunities}
Competitors Analyzed: {len(report.competitor_analyses)}

TOP OPPORTUNITIES
----------------
"""
        
        for i, opp in enumerate(report.missing_entities[:5], 1):
            summary += f"{i}. {opp.name} (Score: {opp.relevance_score:.2f})\n"
            summary += f"   Type: {opp.type}\n"
            summary += f"   Reasoning: {opp.reasoning}\n\n"
        
        summary += "KEY RECOMMENDATIONS\n"
        summary += "-------------------\n"
        
        for i, rec in enumerate(report.recommendations[:3], 1):
            summary += f"{i}. {rec.entity_name}\n"
            summary += f"   Section: {rec.section}\n"
            summary += f"   Recommendation: {rec.recommendation}\n"
            summary += f"   Example: {rec.example_text}\n\n"
        
        return summary
    
    def run(self):
        """Run the main application."""
        self.render_header()
        
        if not self.validate_api_key():
            return
        
        self.render_sidebar()
        
        if st.session_state.analysis_report:
            self.render_results()
        else:
            self.render_input_form()

def main():
    """Main application entry point."""
    app = SEOAnalyzerApp()
    app.run()

if __name__ == "__main__":
    main()
