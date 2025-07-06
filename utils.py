import io
from typing import Dict, Any, List
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from urllib.parse import urlparse
import json

def get_competitor_name(url: str, used_domains: Dict[str, int]) -> str:
    """Generate a unique competitor name based on domain."""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.replace('www.', '')
    if domain not in used_domains:
        used_domains[domain] = 0
    used_domains[domain] += 1
    if used_domains[domain] > 1:
        return f"comp_{domain}_{used_domains[domain]}"
    return f"comp_{domain}"

def validate_urls(urls: List[str]) -> bool:
    """Validate list of URLs."""
    import re
    url_pattern = re.compile(
        r'^https?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    return all(url_pattern.match(url) for url in urls if url)

def create_excel_report(analysis_results: Dict[str, Any]) -> bytes:
    """Create Excel report from analysis results without pandas."""
    output = io.BytesIO()
    workbook = Workbook()
    
    # Remove the default sheet
    if "Sheet" in workbook.sheetnames:
        del workbook["Sheet"]
    
    used_domains = {}
    competitor_names = {}
    if "competitor_urls" in analysis_results:
        for url in analysis_results["competitor_urls"]:
            competitor_names[url] = get_competitor_name(url, used_domains)

    # --- Entity Analysis Sheet ---
    entity_sheet = workbook.create_sheet("Entity Analysis")
    entity_headers = ["Source", "Entity", "Type", "Salience", "Sentiment Score", "Sentiment Magnitude", "Mentions"]
    entity_sheet.append(entity_headers)
    
    entity_data = []
    if "analysis_results" in analysis_results:
        client_analysis = analysis_results["analysis_results"][0]
        for entity_name, data in client_analysis["entities"].items():
            entity_data.append([
                "Client Page - " + analysis_results.get("client_url", ""),
                entity_name,
                data["type"],
                data["salience"],
                data["sentiment"]["score"],
                data["sentiment"]["magnitude"],
                ", ".join([mention["text"] for mention in data["mentions"]])
            ])
        
        if "analysis_results" in analysis_results and len(analysis_results["analysis_results"]) > 1:
            for i, analysis in enumerate(analysis_results["analysis_results"][1:]):
                url = analysis_results["competitor_urls"][i]
                for entity_name, data in analysis["entities"].items():
                    entity_data.append([
                        f"Competitor - {competitor_names[url]}",
                        entity_name,
                        data["type"],
                        data["salience"],
                        data["sentiment"]["score"],
                        data["sentiment"]["magnitude"],
                        ", ".join([mention["text"] for mention in data["mentions"]])
                    ])
    for row in entity_data:
        entity_sheet.append(row)
    adjust_column_width(entity_sheet, entity_headers)
    
    # --- Keyword Analysis Sheet ---
    keyword_sheet = workbook.create_sheet("Keyword Analysis")
    
    keyword_headers = ["Source", "Keyword", "Density", "Count", "TF-IDF", "Phrase Count"]
    keyword_sheet.append(keyword_headers)
    
    keyword_data = []
    if "analysis_results" in analysis_results:
        client_analysis = analysis_results["analysis_results"][0]
        for keyword, data in client_analysis["keyword_analysis"].items():
            keyword_data.append([
                "Client Page - " + analysis_results.get("client_url", ""),
                keyword,
                data.get("density", 0),
                data.get("count", 0),
                data.get("tf_idf", 0),
                data.get("phrase_counts", 0)
            ])
        if "analysis_results" in analysis_results and len(analysis_results["analysis_results"]) > 1:
            for i, analysis in enumerate(analysis_results["analysis_results"][1:]):
                url = analysis_results["competitor_urls"][i]
                for keyword, data in analysis["keyword_analysis"].items():
                    keyword_data.append([
                        f"Competitor - {competitor_names[url]}",
                        keyword,
                        data.get("density", 0),
                        data.get("count", 0),
                        data.get("tf_idf", 0),
                        data.get("phrase_counts", 0)
                    ])
    for row in keyword_data:
        keyword_sheet.append(row)
    adjust_column_width(keyword_sheet, keyword_headers)
    
    # --- Missing Entities Sheet ---
    missing_entities_sheet = workbook.create_sheet("Missing Entities")
    missing_entities_headers = ["Entity", "Type"]
    missing_entities_sheet.append(missing_entities_headers)
    
    missing_entities = []
    if "comparison_results" in analysis_results and "missing_entities" in analysis_results["comparison_results"]:
        for entity_name, data in analysis_results["comparison_results"]["missing_entities"].items():
            missing_entities.append([
                entity_name,
                data["type"]
            ])
    for row in missing_entities:
        missing_entities_sheet.append(row)
    adjust_column_width(missing_entities_sheet, missing_entities_headers)
    
    
    # --- Document Sentiment Sheet ---
    sentiment_sheet = workbook.create_sheet("Document Sentiment")
    sentiment_headers = ["Source", "Score", "Magnitude"]
    sentiment_sheet.append(sentiment_headers)
    
    sentiment_data = []
    if "analysis_results" in analysis_results:
        client_analysis = analysis_results["analysis_results"][0]
        sentiment_data.append([
            "Client Page - " + analysis_results.get("client_url", ""),
            client_analysis["document_sentiment"]["score"],
            client_analysis["document_sentiment"]["magnitude"]
        ])
        if "analysis_results" in analysis_results and len(analysis_results["analysis_results"]) > 1:
            for i, analysis in enumerate(analysis_results["analysis_results"][1:]):
                url = analysis_results["competitor_urls"][i]
                sentiment_data.append([
                    f"Competitor - {competitor_names[url]}",
                    analysis["document_sentiment"]["score"],
                    analysis["document_sentiment"]["magnitude"]
                ])
    for row in sentiment_data:
        sentiment_sheet.append(row)
        
    adjust_column_width(sentiment_sheet, sentiment_headers)
    
    workbook.save(output)
    output.seek(0)
    return output.getvalue()

def adjust_column_width(sheet, headers):
    for col_num, header in enumerate(headers, 1):
        column_letter = get_column_letter(col_num)
        max_length = max(len(str(header)), max(len(str(cell.value)) if cell.value else 0 for cell in sheet[column_letter]))
        adjusted_width = (max_length + 2)
        adjusted_width = 75 if adjusted_width > 75 else adjusted_width # if adjusted_width > 75, make it 75 to prevent column width from being too large
        sheet.column_dimensions[column_letter].width = adjusted_width