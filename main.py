import os
import asyncio
import aiohttp
from loguru import logger
from typing import List
from datetime import datetime
from collections import defaultdict

from utils import get_competitor_name, validate_urls
from data_models import FinalState
from entity_analysis import (
    analyze_content, scrape_content, compare_pages, 
    select_entities_for_integration, generate_entity_recommendations
)

async def main_analysis(client_url: str, competitor_urls: List[str], credentials_path: str, output_folder: str):
    """Performs the entire content analysis and generates a report."""
    try:
        logger.info("Starting analysis process...")

        if not validate_urls([client_url] + competitor_urls):
            logger.error("Invalid URL format. Please enter valid URLs.")
            return

        logger.info("Initializing domain counter and competitor mapping...")
        used_domains = defaultdict(int)
        competitor_names = {url: get_competitor_name(url, used_domains) for url in competitor_urls}

        async with aiohttp.ClientSession() as session:
            # Scrape content
            logger.info("Scraping content from URLs...")
            scrape_tasks = [scrape_content(client_url, session)] + [scrape_content(url, session) for url in competitor_urls]
            scrape_results = await asyncio.gather(*scrape_tasks)

            client_content = scrape_results[0]
            if not client_content:
                logger.error("Failed to scrape client page.")
                return

            competitive_contents = [content for content in scrape_results[1:] if content]
            valid_competitor_urls = [url for url, content in zip(competitor_urls, scrape_results[1:]) if content]

            all_documents = [client_content] + competitive_contents
            
            # Analyze content
            logger.info("Analyzing scraped content...")
            analysis_tasks = [analyze_content(client_content, credentials_path, all_documents)] + [
                analyze_content(content, credentials_path, all_documents) for content in competitive_contents
            ]
            analysis_results = await asyncio.gather(*analysis_tasks)

            client_analysis = analysis_results[0]
            competitive_analyses = analysis_results[1:]

            # Compare results
            logger.info("Comparing client content with competitors...")
            comparison_results = compare_pages(client_analysis, competitive_analyses, valid_competitor_urls)

            # Update competitor names in the results
            for analysis_type in ["missing_entities", "missing_keywords"]:
                for item_name, item_data in comparison_results.get(analysis_type, {}).items():
                    item_data["competitors"] = {competitor_names[url]: metrics for url, metrics in item_data["competitors"].items()}

            # Select entities for integration
            logger.info("Selecting entities for integration...")
            selected_entities_response = await select_entities_for_integration(comparison_results.get("missing_entities", {}))
            selected_entities = selected_entities_response.selected_entities

            # Generate recommendations
            logger.info("Generating recommendations for selected entities...")
            recommendation_tasks = [generate_entity_recommendations(entity_selection, client_content) for entity_selection in selected_entities]
            recommendations = await asyncio.gather(*recommendation_tasks)

            # Export results
            logger.info("Creating analysis report...")
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            final_state = FinalState(
                client_url=client_url,
                competitor_urls=valid_competitor_urls,
                analysis_results=analysis_results,
                comparison_results=comparison_results,
                selected_entities=selected_entities_response,
                recommendation_overview=recommendations
            )
            
            markdown = final_state.to_markdown
            excel = final_state.to_excel

            md_filename = f"{current_time}_analysis_report.md"
            excel_filename = f"{current_time}_analysis_data.xlsx"
            
            os.makedirs(output_folder, exist_ok=True)
            output_path_md = os.path.join(output_folder, md_filename)
            output_path_excel = os.path.join(output_folder, excel_filename)
            
            with open(output_path_md, "w") as file:
                file.write(markdown)
            
            logger.info(f"Markdown report saved to: {output_path_md}")
            
            with open(output_path_excel, "wb") as file:
                file.write(excel)

            logger.info(f"Excel report saved to: {output_path_excel}")
            logger.info("Analysis process completed successfully")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    urls = {
        "client_url": "https://www.nerdwallet.com/article/taxes/bonus-tax-rate-how-are-bonuses-taxed",
        "competitor_urls": [
            "https://www.fidelity.com/learning-center/smart-money/bonus-tax-rate",
            "https://www.oysterhr.com/library/how-are-bonuses-taxed",
            "https://www.bankrate.com/taxes/how-bonuses-are-taxed/"
        ]
    }
    asyncio.run(main_analysis(urls["client_url"], urls["competitor_urls"], "./service_account.json", "output"))
