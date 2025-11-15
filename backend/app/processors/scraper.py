"""
Web scraping processor for data ingestion.
"""
import asyncio
from typing import Dict, Any, Optional, List
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from app.core.logging import get_logger
from app.models.schemas.ingestion import ScrapingConfig

logger = get_logger(__name__)


class WebScraper:
    """Web scraper for data extraction."""
    
    DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; AnalyticsPlatform/1.0; +https://example.com/bot)"
    DEFAULT_TIMEOUT = 30.0
    
    def __init__(self):
        self.logger = logger
        self._robots_cache: Dict[str, Optional[RobotFileParser]] = {}
    
    def _check_robots_txt(self, url: str) -> bool:
        """
        Check if URL is allowed by robots.txt.
        
        Args:
            url: URL to check
            
        Returns:
            True if allowed, False otherwise
        """
        try:
            parsed = urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            
            if base_url not in self._robots_cache:
                robots_url = urljoin(base_url, "/robots.txt")
                rp = RobotFileParser()
                rp.set_url(robots_url)
                try:
                    rp.read()
                    self._robots_cache[base_url] = rp
                except Exception as e:
                    logger.warning(f"Could not read robots.txt from {robots_url}: {e}")
                    self._robots_cache[base_url] = None
            
            rp = self._robots_cache[base_url]
            if rp is None:
                return True  # If robots.txt is not accessible, allow by default
            
            return rp.can_fetch(self.DEFAULT_USER_AGENT, url)
            
        except Exception as e:
            logger.warning(f"Error checking robots.txt for {url}: {e}")
            return True  # Allow by default if check fails
    
    async def scrape_page(
        self,
        url: str,
        selectors: Dict[str, str],
        respect_robots: bool = True,
        timeout: float = DEFAULT_TIMEOUT
    ) -> Dict[str, Any]:
        """
        Scrape a single page.
        
        Args:
            url: URL to scrape
            selectors: CSS selectors for data extraction
            respect_robots: Whether to respect robots.txt
            timeout: Request timeout
            
        Returns:
            Dictionary with extracted data
        """
        # Check robots.txt
        if respect_robots and not self._check_robots_txt(url):
            raise ValueError(f"URL {url} is disallowed by robots.txt")
        
        try:
            headers = {
                "User-Agent": self.DEFAULT_USER_AGENT
            }
            
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract data using selectors
                extracted_data = {}
                for key, selector in selectors.items():
                    elements = soup.select(selector)
                    if elements:
                        # Extract text from all matching elements
                        extracted_data[key] = [elem.get_text(strip=True) for elem in elements]
                    else:
                        extracted_data[key] = []
                
                return {
                    "url": url,
                    "data": extracted_data,
                    "timestamp": asyncio.get_event_loop().time()
                }
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error scraping {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            raise
    
    async def scrape(
        self,
        config: ScrapingConfig,
        max_pages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Scrape data based on configuration.
        
        Args:
            config: Scraping configuration
            max_pages: Maximum number of pages to scrape
            
        Returns:
            List of scraped data dictionaries
        """
        results = []
        pages_to_scrape = [str(config.url)]
        
        # Handle pagination if configured
        if config.pagination:
            # Simple pagination support - can be extended
            max_pages = max_pages or config.max_pages or 1
            # For now, we'll just scrape the initial URL
            # Full pagination support would require more complex logic
        
        max_pages = max_pages or config.max_pages or 1
        
        for i, url in enumerate(pages_to_scrape[:max_pages]):
            try:
                # Rate limiting
                if i > 0 and config.rate_limit:
                    await asyncio.sleep(config.rate_limit)
                
                page_data = await self.scrape_page(
                    url,
                    config.selectors,
                    config.respect_robots_txt
                )
                results.append(page_data)
                
            except Exception as e:
                logger.error(f"Error scraping page {url}: {e}")
                # Continue with next page even if one fails
                continue
        
        return results

