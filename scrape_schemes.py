# scrape_schemes.py
# -*- coding: utf-8 -*-

# Purpose: Scrapes scheme data from myscheme.gov.in
#          Uses Selenium for link collection (search pages)
#          Uses Requests for detail page extraction (generic content)

import requests
from bs4 import BeautifulSoup, NavigableString
import time
import logging
from urllib.parse import urljoin
import json
import os
import sys

# --- Selenium Imports ---
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException, ElementClickInterceptedException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("WARNING: Selenium libraries not found. Scraping functionality will be limited.")
    print("         Install them using: pip install selenium webdriver-manager")

# ==================================
# ---        CONFIGURATION       ---
# ==================================
BASE_URL = "https://www.myscheme.gov.in"
HEADERS = { # Used for Requests
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9'
}
REQUEST_DELAY = 1.5  # Delay between requests calls (seconds)
REQUEST_TIMEOUT = 30 # Timeout for requests calls (seconds)

TARGET_SCHEME_COUNT = 100
MAX_SEARCH_PAGES = 15 # Max search result pages to attempt pagination on

# Selectors for Selenium link extraction
SEARCH_RESULTS_LINK_SELECTOR = 'h2 > a[href^="/schemes/"]' # Links inside H2
PAGE_NUMBER_LINK_XPATH = "//ul[contains(@class, 'list-none')]//a[normalize-space()='{}']" # Placeholder for page number <a> tag
PAGE_NUMBER_LI_XPATH = "//ul[contains(@class, 'list-none')]//li[normalize-space()='{}']" # Placeholder for page number <li> tag

SEARCH_PAGE_WAIT_TIMEOUT = 25 # Wait time for search page dynamic content to appear
PAGINATION_CLICK_WAIT = 10    # Max wait time to find/click pagination element
PAGINATION_LOAD_DELAY = 3.0 # Static delay after clicking pagination (increase if pages don't load)

# Ensure data directory exists
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_JSON_FILE = os.path.join(OUTPUT_DIR, f"myscheme_{TARGET_SCHEME_COUNT}_schemes_generic.json")

# ==================================
# ---        LOGGING SETUP       ---
# ==================================
log_format = '%(asctime)s - %(levelname)s - %(funcName)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format,
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# ==================================
# ---  SELENIUM HELPER FUNCTIONS ---
# ==================================
# <Copy the final initialize_driver() function here>
def initialize_driver():
    """Initializes and returns a Selenium WebDriver instance."""
    if not SELENIUM_AVAILABLE: return None
    driver = None
    logger.info("Initializing Selenium WebDriver...")
    try:
        logger.info("Setting up Chrome options...")
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument(f"user-agent={HEADERS['User-Agent']}")

        logger.info("Initializing Chrome WebDriver service using webdriver-manager...")
        service = ChromeService(ChromeDriverManager().install())

        logger.info("Creating WebDriver instance...")
        driver = webdriver.Chrome(service=service, options=chrome_options)
        logger.info("WebDriver initialized successfully.")
    except WebDriverException as e:
        logger.error(f"WebDriverException occurred during initialization: {e}")
        print("Please ensure Google Chrome is installed and accessible.")
        driver = None
    except Exception as e:
        logger.error(f"An unexpected error occurred during WebDriver setup: {e}", exc_info=True)
        if driver:
            try: driver.quit()
            except Exception: pass
        driver = None
    return driver

# <Copy the get_links_from_page_selenium() function definition here - modified to not need fetch>
# This version assumes the driver is already on the correct page
def extract_links_from_current_page(driver_instance):
    """Extracts scheme links from the current page source in the driver."""
    if not driver_instance: return []
    logger.info("Extracting links from current Selenium page source...")
    try:
        page_source = driver_instance.page_source
        soup = BeautifulSoup(page_source, 'lxml')
    except Exception as e:
        logger.error(f"Failed to get/parse current page source: {e}", exc_info=True)
        return []

    selector = SEARCH_RESULTS_LINK_SELECTOR
    link_elements = soup.select(selector)
    logger.info(f"Found {len(link_elements)} link elements using selector '{selector}'.")

    scheme_links_found = set()
    for link_tag in link_elements:
        relative_url = link_tag.get('href')
        if relative_url and relative_url.startswith("/schemes/"):
            absolute_url = urljoin(BASE_URL, relative_url)
            if absolute_url != urljoin(BASE_URL, "/schemes/"):
                scheme_links_found.add(absolute_url)
    unique_links = list(scheme_links_found)
    logger.debug(f"Extracted {len(unique_links)} unique links from current page.")
    return unique_links


# ==================================
# --- REQUESTS HELPER FUNCTIONS  ---
# ==================================
# <Copy the fetch_html_requests() function definition here>
def fetch_html_requests(url):
    """Fetches HTML using requests."""
    logger.debug(f"Attempting to fetch (requests): {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        logger.info(f"Successfully fetched (requests) {url} (Status: {response.status_code})")
        time.sleep(REQUEST_DELAY)
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Requests error fetching {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during requests fetch for {url}: {e}", exc_info=True)
        return None

# <Copy the extract_generic_content() function definition here>
# <Also copy its helper extract_text_or_default() if used>
def extract_text_or_default(element, selector, default="N/A"):
    """Safely extracts text from a BeautifulSoup element using a selector."""
    if not element: return default
    try:
        found = element.select_one(selector)
        if found:
            text = found.get_text(strip=True)
            return text if text else default
        return default
    except Exception as e:
        logger.error(f"Error extracting text with selector '{selector}': {e}")
        return default

def extract_generic_content(scheme_url):
    """ Fetches (requests) and extracts generic content from a scheme detail page."""
    logger.info(f"--- Extracting generic content (requests) for: {scheme_url} ---")
    html = fetch_html_requests(scheme_url)
    if not html: return None

    try:
        soup = BeautifulSoup(html, 'lxml')
    except Exception as e:
        logger.error(f"Failed to parse detail page HTML for {scheme_url}: {e}", exc_info=True)
        return None

    main_content_area = soup.select_one('div#scrollDiv, main, article, div[role="main"]') or soup.body
    if not main_content_area:
        logger.error(f"Could not find main content area for {scheme_url}")
        return None

    page_title = soup.title.string.strip() if soup.title else "N/A"
    main_h1_tag = main_content_area.select_one('h1[title]')
    main_heading = main_h1_tag.get_text(strip=True) if main_h1_tag else extract_text_or_default(main_content_area,'h1, h2')

    sections = []
    headings = main_content_area.find_all(['h3', 'h2', 'h4'], class_=lambda x: x and 'font-heading' in x)
    if not headings: headings = main_content_area.find_all(['h1', 'h2', 'h3', 'h4'])
    processed_headings = set()

    for heading in headings:
        heading_text = heading.get_text(strip=True)
        if not heading_text or heading_text in processed_headings: continue
        processed_headings.add(heading_text)
        heading_container = heading.find_parent(['a', 'div']) or heading
        content_element = None
        current = heading_container
        while current:
            current = current.find_next_sibling()
            if current and current.name:
                 if current.name == 'div' and ('grid' in current.get('class', []) or 'markdown-options' in current.get('class', []) or not current.has_attr('id')):
                      content_element = current
                      break
                 elif current.name in ['h1', 'h2', 'h3', 'h4'] or (current.name == 'div' and current.has_attr('id') and current.has_attr('class') and 'pt-10' in current.get('class',[])):
                      break
        combined_content = "N/A"
        if content_element:
            target = content_element.select_one('div.markdown-options') or content_element
            combined_content = target.get_text(separator='\n', strip=True)
        if combined_content and combined_content != "N/A":
             sections.append({'heading_tag': heading.name, 'heading_text': heading_text, 'content': combined_content})

    links = []
    all_link_tags = main_content_area.find_all('a', href=True)
    for link_tag in all_link_tags:
        href = link_tag['href']
        text = link_tag.get_text(strip=True)
        if href and not href.startswith(('#', 'javascript:')):
            absolute_url = urljoin(BASE_URL, href)
            links.append({'text': text, 'url': absolute_url})

    extracted_data = {
        'scheme_url': scheme_url, 'page_title': page_title, 'main_heading_h1': main_heading,
        'extracted_sections': sections, 'extracted_links': links
    }
    logger.info(f"--- Finished generic extraction for: {scheme_url} ---")
    return extracted_data

# ==================================
# ---     MAIN SCRAPING LOGIC    ---
# ==================================
def run_scraper():
    """Runs the full scraping process."""
    if not SELENIUM_AVAILABLE:
        logger.error("Selenium is not installed. Cannot run the scraper.")
        sys.exit(1)

    logger.info("--- Starting MyScheme Scraper ---")
    driver = None
    all_scheme_urls = set()
    all_generic_data = []
    processed_pages = 0
    max_page_to_reach = (TARGET_SCHEME_COUNT + 9) // 10 + 1
    if max_page_to_reach > MAX_SEARCH_PAGES: max_page_to_reach = MAX_SEARCH_PAGES

    try:
        driver = initialize_driver()
        if not driver: raise Exception("Failed to initialize Selenium WebDriver. Exiting.")

        logger.info(f"Attempting to collect up to {TARGET_SCHEME_COUNT} scheme links...")
        current_page_url = f"{BASE_URL}/search?page=1"
        driver.get(current_page_url)

        for page_num in range(1, max_page_to_reach + 1):
            logger.info(f"Processing search page {page_num} (URL: {driver.current_url})")
            try:
                WebDriverWait(driver, SEARCH_PAGE_WAIT_TIMEOUT).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, SEARCH_RESULTS_LINK_SELECTOR))
                )
                logger.info(f"Links found/reloaded on page {page_num}.")
                time.sleep(1.0)
            except TimeoutException:
                logger.warning(f"Timeout waiting for links on page {page_num}.")
                break

            links_on_page = extract_links_from_current_page(driver)
            if not links_on_page and page_num > 1:
                 logger.warning(f"No links extracted on page {page_num}.")
                 break

            newly_found_count = len(links_on_page.difference(all_scheme_urls))
            all_scheme_urls.update(links_on_page)
            processed_pages += 1
            logger.info(f"Page {page_num}: Found {len(links_on_page)} links ({newly_found_count} new). Total unique links: {len(all_scheme_urls)}")

            if len(all_scheme_urls) >= TARGET_SCHEME_COUNT or page_num == max_page_to_reach: break
            if newly_found_count == 0 and page_num > 1:
                 logger.info("No new links found. Assuming end.")
                 break

            next_page_num = page_num + 1
            try:
                logger.info(f"Looking for link/element for page number: {next_page_num}...")
                page_xpath = PAGE_NUMBER_LINK_XPATH.format(next_page_num) # Prefer link
                page_element = None
                try:
                     page_element = WebDriverWait(driver, PAGINATION_CLICK_WAIT).until(
                          EC.element_to_be_clickable((By.XPATH, page_xpath))
                     )
                     logger.info(f"Found clickable 'a' tag for page {next_page_num}.")
                except TimeoutException:
                     logger.warning(f"No clickable 'a' tag found for page {next_page_num}, trying 'li'...")
                     page_xpath = PAGE_NUMBER_LI_XPATH.format(next_page_num) # Fallback to li
                     try:
                          page_element = WebDriverWait(driver, 5).until(
                               EC.element_to_be_clickable((By.XPATH, page_xpath))
                          )
                          logger.info(f"Found clickable 'li' tag for page {next_page_num}.")
                     except TimeoutException:
                          logger.error(f"Could not find clickable element for page {next_page_num}. Ending pagination.")
                          break

                if page_element:
                    logger.info(f"Clicking element for page {next_page_num}...")
                    try: page_element.click()
                    except ElementClickInterceptedException:
                         logger.warning(f"Direct click intercepted for page {next_page_num}, trying JS click.")
                         driver.execute_script("arguments[0].click();", page_element)
                    time.sleep(PAGINATION_LOAD_DELAY)
            except Exception as e_next:
                 logger.error(f"Error finding or clicking for page {next_page_num}: {e_next}")
                 break

        final_urls_to_scrape = list(all_scheme_urls)[:TARGET_SCHEME_COUNT]
        logger.info(f"Collected {len(final_urls_to_scrape)} unique scheme URLs after checking {processed_pages} pages.")
        if not final_urls_to_scrape: raise Exception("No scheme URLs collected.")

        logger.info(f"--- Starting detail scraping loop for {len(final_urls_to_scrape)} schemes ---")
        for i, scheme_url in enumerate(final_urls_to_scrape):
            logger.info(f"Processing scheme {i+1}/{len(final_urls_to_scrape)}: {scheme_url}")
            details = extract_generic_content(scheme_url)
            if details: all_generic_data.append(details)
            else: logger.warning(f"Extraction returned None for scheme {i+1}: {scheme_url}")
        logger.info(f"--- Finished detail scraping loop ---")
        logger.info(f"Successfully extracted data for {len(all_generic_data)} out of {len(final_urls_to_scrape)} schemes attempted.")

        if all_generic_data:
            logger.info(f"Saving extracted data to {OUTPUT_JSON_FILE}...")
            try:
                with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
                    json.dump(all_generic_data, f, ensure_ascii=False, indent=4)
                logger.info(f"Data successfully saved.")
                print(f"\n---> Scraped data for {len(all_generic_data)} schemes saved to {OUTPUT_JSON_FILE} <---")
            except Exception as e:
                logger.error(f"Failed to save data: {e}", exc_info=True)
        else: logger.warning("No data collected to save.")

    except Exception as main_error:
        logger.error(f"Scraper failed: {main_error}", exc_info=True)
    finally:
        if driver:
            logger.info("Quitting Selenium WebDriver...")
            try: driver.quit()
            except Exception as e: logger.error(f"Error quitting WebDriver: {e}")
            logger.info("WebDriver quit.")
    logger.info("--- MyScheme Scraper Finished ---")

if __name__ == "__main__":
    run_scraper()