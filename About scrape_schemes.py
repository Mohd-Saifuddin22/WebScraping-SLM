The goal of this phase was to extract detailed information for approximately 100 unique government schemes from the official portal.

Challenges & Approach:

Dynamic Content: The primary challenge on the scheme listing/search pages (/search?page=...) was dynamic content loading. The list of schemes appeared to be rendered using JavaScript after the initial page load. Standard fetching with requests failed to capture these scheme links as it only receives the initial HTML source.
Solution: We employed Selenium with webdriver-manager to automate a headless Chrome browser. Selenium loads the search page, waits for the scheme links (specifically h2 > a[href^="/schemes/"]) to appear in the DOM (using WebDriverWait), ensuring JavaScript rendering completes before extracting the links.

Pagination: The pagination control at the bottom of the search results did not use a standard "Next" button. Instead, it used clickable page number elements (likely <li> or <a> tags containing the number) and SVG icons for previous/next arrows. Directly clicking the SVG icon proved unreliable (intercepted clicks, no JS .click() method).
Solution: The final script uses Selenium to iteratively find and click the specific element corresponding to the next page number (e.g., clicking "2", then "3", etc.). It includes fallbacks (trying both <a> and <li> tags via XPath) and uses JavaScript clicks when direct clicks are intercepted. The loop continues until the target number of unique scheme URLs (e.g., 100) is collected or pagination ends.

Detail Page Extraction: Initial analysis suggested some detail pages might also use dynamic content (like tabs). However, testing revealed that many detail pages could be adequately processed using the faster requests library to fetch the static HTML.
Solution: For efficiency, the script fetches individual scheme detail pages using requests. It then parses the HTML using BeautifulSoup4 (with the lxml parser).

Generic Content Extraction (extract_generic_content): Instead of relying on highly specific selectors for predefined fields (which proved brittle across different schemes), a more robust generic approach was adopted. This function identifies the main content area, finds major headings (h1-h4), and extracts the block of text/list/table content that logically follows each heading by analyzing the sibling structure (handling cases where headings are nested within links). This captures the core information in a {'heading': ..., 'content': ...} format. Scheme Name and Ministry are extracted using dedicated logic based on common patterns (e.g., h1[title], h3 preceding h1). Links within the main content are also extracted.

Output:The scraper (scrape_schemes.py) generates the data/myscheme_100_schemes_generic.json file. This is a list of dictionaries, where each dictionary represents a scraped scheme and contains its URL, title, main heading, a list of extracted sections (heading + content), and a list of extracted links.
