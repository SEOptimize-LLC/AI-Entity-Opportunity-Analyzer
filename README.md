# AI Entity Opportunity Analyzer

This Python application, `ai_entity_opportunity_analyzer`, is designed to analyze web page content, identify SEO opportunities, and provide actionable recommendations using the Google Cloud Natural Language API and AI-powered insights. It compares a client's web page with competitor pages to identify missing entities and keywords, then uses an LLM to select the most relevant entities and generate recommendations for integration.

## Features

- **Asynchronous Web Scraping:** Extracts text content from web pages using `aiohttp` and `BeautifulSoup4` for efficient and concurrent scraping.
- **Google Cloud Natural Language API:** Leverages the Google Cloud Natural Language API for advanced text analysis, including entity recognition, sentiment analysis, and keyword extraction.
- **Keyword and Phrase Analysis:** Identifies key terms and phrases, calculating TF-IDF and density.
- **Competitive Analysis:** Compares client page analysis with competitor analyses to identify missing entities and keywords.
- **AI-Powered Entity Selection:** Uses an LLM to select the top 10 most relevant missing entities for integration, considering relevance scores and reasoning.
- **AI-Powered Recommendation Generation:** Generates structured recommendations for each selected entity, providing specific guidance on how to integrate them effectively.
- **Structured Reporting:** Generates an Excel report to capture the analysis, including entity analysis, keyword analysis, missing entities/keywords, and AI-powered recommendations.
- **Logging:** Includes detailed logging using `loguru` for monitoring and debugging.

## Files

- `entity_analysis.py`: Contains the core logic for scraping, text preprocessing, calling the Google Natural Language API, and generating AI-powered recommendations.
- `main.py`: Contains the main logic for running the analysis, generating reports, and handling inputs and errors.
- `utils.py`: Contains utility functions for creating Excel reports and handling competitor names.
- `data_models.py`: Contains Pydantic data models for structuring the data used in the application.
- `service_account.json`: This file contains the service account json credentials for GCP (Google Cloud Platform) that will need to have Google Cloud Natural Language API activated.
- `.env`: This file contains the API Key for OpenAI. (Not included in repo)

## Setup

1.  **Clone the repository:**

    ```bash
    git clone [repository URL]
    cd ai_entity_opportunity_analyzer
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    \*Note: the requirements.txt file is created by running `pip freeze > requirements.txt`

3.  **Google Cloud API Setup:**
    - Enable the Google Cloud Natural Language API in your Google Cloud project.
    - Create a service account and download the JSON key file.
    - Rename the downloaded key file to `service_account.json` and place it in the root of your project directory (the same level as `main.py`).
4.  **OpenAI API Setup:**
    - Create an account with OpenAI and generate an API key.
    - Create a `.env` file in the root of your project directory (the same level as `main.py`).
    - Add the following line to the `.env` file, replacing `YOUR_OPENAI_API_KEY` with your actual API key:
      ```
      OPENAI_API_KEY=YOUR_OPENAI_API_KEY
      ```
5.  **Environment Variable**:
    - The path to `service_account.json` is automatically handled by `analyze_content`.
    - No need to set the GOOGLE_APPLICATION_CREDENTIALS environment variable explicitly.

## Usage

To run the analysis, execute `main.py`:

```bash
python main.py
```

- `main.py` is designed with parameters that accept a dictionary containing client and competitor URLs.
- The parameters are stored as a global variable `urls` in `main.py`.
- The output of the analysis will be saved to an output folder, named `output`, created in the directory `main.py` is located in.

## Input Format

The application takes URLs as input.

- The `urls` variable is defined in `main.py`.
- Example:

  ```python
  urls = {
      "client_url": "https://turbotax.intuit.com/tax-tips/jobs-and-career/how-bonuses-are-taxed/L7UjtAZbh",
      "competitor_urls": [
          "https://www.fidelity.com/learning-center/smart-money/bonus-tax-rate",
          "https://www.oysterhr.com/library/how-are-bonuses-taxed",
          "https://www.bankrate.com/taxes/how-bonuses-are-taxed/"
      ]
  }
  ```

## Output

- The application generates an Excel file, `analysis_report_<date-time>.xlsx`, with the analysis results, located in the `output` folder.
- The Excel file includes the analysis for each page, a comparison of missing entities/keywords, and AI-powered recommendations.
- Analysis is done for the client URL, its competitive URLs, and combined analysis across all pages.
