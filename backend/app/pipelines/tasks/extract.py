"""
Extract tasks for ETL pipeline.
"""
from typing import Dict, Any, List
import pandas as pd
import asyncio

from prefect import task
from app.core.logging import get_logger
from app.processors.csv_processor import CSVProcessor
from app.processors.api_processor import APIProcessor
from app.processors.scraper import WebScraper
from app.models.database.data_sources import DataSourceType
from app.storage.cloud_storage import cloud_storage

logger = get_logger(__name__)


async def _extract_csv_impl(
    source_id: int,
    file_path: str,
    encoding: str = "utf-8"
) -> pd.DataFrame:
    """
    Internal implementation for CSV extraction (not a Prefect task).
    
    Args:
        source_id: Data source ID
        file_path: Path to CSV file (can be cloud storage path)
        encoding: File encoding
        
    Returns:
        pandas DataFrame
    """
    logger.info(f"Extracting CSV data from {file_path} for source {source_id}")
    
    try:
        # Check if file is in cloud storage
        if file_path.startswith("s3://") or file_path.startswith("https://"):
            # Download from cloud storage
            file_content = await cloud_storage.download_file(file_path)
        else:
            # Read from local file
            with open(file_path, "rb") as f:
                file_content = f.read()
        
        processor = CSVProcessor()
        df = processor.read_csv(file_content, encoding=encoding)
        
        logger.info(f"Successfully extracted {len(df)} rows from CSV")
        return df
        
    except Exception as e:
        logger.error(f"Error extracting CSV data: {e}")
        raise


@task(name="extract_csv", retries=2, retry_delay_seconds=5)
async def extract_csv(
    source_id: int,
    file_path: str,
    encoding: str = "utf-8"
) -> pd.DataFrame:
    """
    Extract data from CSV file (Prefect task wrapper).
    
    Args:
        source_id: Data source ID
        file_path: Path to CSV file (can be cloud storage path)
        encoding: File encoding
        
    Returns:
        pandas DataFrame
    """
    return await _extract_csv_impl(source_id, file_path, encoding)


async def _extract_api_impl(
    source_id: int,
    url: str,
    method: str = "GET",
    headers: Dict[str, str] = None,
    params: Dict[str, Any] = None,
    body: Dict[str, Any] = None,
    auth_config: Dict[str, Any] = None,
    rate_limit: int = None,
    retry_count: int = 3
) -> Dict[str, Any]:
    """
    Internal implementation for API extraction (not a Prefect task).
    
    Args:
        source_id: Data source ID
        url: API endpoint URL
        method: HTTP method
        headers: HTTP headers
        params: Query parameters
        body: Request body
        auth_config: Authentication configuration
        rate_limit: Rate limit (requests per minute)
        retry_count: Number of retry attempts
        
    Returns:
        Dictionary with API response data
    """
    logger.info(f"Extracting data from API {url} for source {source_id}")
    
    try:
        from app.models.schemas.ingestion import APIAuthConfig
        
        auth = None
        if auth_config:
            auth = APIAuthConfig(**auth_config)
        
        processor = APIProcessor()
        result = await processor.fetch_data(
            url=url,
            method=method,
            headers=headers,
            params=params,
            body=body,
            auth=auth,
            rate_limit=rate_limit,
            retry_count=retry_count
        )
        
        logger.info(f"Successfully extracted data from API")
        return result
        
    except Exception as e:
        logger.error(f"Error extracting API data: {e}")
        raise


@task(name="extract_api", retries=3, retry_delay_seconds=10)
async def extract_api(
    source_id: int,
    url: str,
    method: str = "GET",
    headers: Dict[str, str] = None,
    params: Dict[str, Any] = None,
    body: Dict[str, Any] = None,
    auth_config: Dict[str, Any] = None,
    rate_limit: int = None,
    retry_count: int = 3
) -> Dict[str, Any]:
    """
    Extract data from API (Prefect task wrapper).
    
    Args:
        source_id: Data source ID
        url: API endpoint URL
        method: HTTP method
        headers: HTTP headers
        params: Query parameters
        body: Request body
        auth_config: Authentication configuration
        rate_limit: Rate limit (requests per minute)
        retry_count: Number of retry attempts
        
    Returns:
        Dictionary with API response data
    """
    return await _extract_api_impl(source_id, url, method, headers, params, body, auth_config, rate_limit, retry_count)


async def _extract_web_scrape_impl(
    source_id: int,
    url: str,
    selectors: Dict[str, str],
    respect_robots: bool = True,
    rate_limit: int = 1,
    max_pages: int = 1
) -> List[Dict[str, Any]]:
    """
    Internal implementation for web scraping (not a Prefect task).
    
    Args:
        source_id: Data source ID
        url: URL to scrape
        selectors: CSS selectors for data extraction
        respect_robots: Whether to respect robots.txt
        rate_limit: Delay between requests (seconds)
        max_pages: Maximum pages to scrape
        
    Returns:
        List of scraped data dictionaries
    """
    logger.info(f"Extracting data from web scraping {url} for source {source_id}")
    
    try:
        from app.models.schemas.ingestion import ScrapingConfig
        
        config = ScrapingConfig(
            url=url,
            selectors=selectors,
            respect_robots_txt=respect_robots,
            rate_limit=rate_limit,
            max_pages=max_pages
        )
        
        scraper = WebScraper()
        results = await scraper.scrape(config, max_pages)
        
        logger.info(f"Successfully scraped {len(results)} pages")
        return results
        
    except Exception as e:
        logger.error(f"Error extracting web scraped data: {e}")
        raise


@task(name="extract_web_scrape", retries=2, retry_delay_seconds=10)
async def extract_web_scrape(
    source_id: int,
    url: str,
    selectors: Dict[str, str],
    respect_robots: bool = True,
    rate_limit: int = 1,
    max_pages: int = 1
) -> List[Dict[str, Any]]:
    """
    Extract data from web scraping (Prefect task wrapper).
    
    Args:
        source_id: Data source ID
        url: URL to scrape
        selectors: CSS selectors for data extraction
        respect_robots: Whether to respect robots.txt
        rate_limit: Delay between requests (seconds)
        max_pages: Maximum pages to scrape
        
    Returns:
        List of scraped data dictionaries
    """
    return await _extract_web_scrape_impl(source_id, url, selectors, respect_robots, rate_limit, max_pages)


@task(name="extract_data")
async def extract_data(
    source_type: DataSourceType,
    source_id: int,
    config: Dict[str, Any]
) -> Any:
    """
    Generic extract task that routes to appropriate extractor.
    
    Args:
        source_type: Type of data source
        source_id: Data source ID
        config: Source-specific configuration
        
    Returns:
        Extracted data (format depends on source type)
    """
    logger.info(f"Extracting data for source {source_id} (type: {source_type})")
    
    if source_type == DataSourceType.CSV:
        return await _extract_csv_impl(
            source_id=source_id,
            file_path=config.get("file_path"),
            encoding=config.get("encoding", "utf-8")
        )
    
    elif source_type == DataSourceType.API:
        return await _extract_api_impl(
            source_id=source_id,
            url=config.get("url"),
            method=config.get("method", "GET"),
            headers=config.get("headers"),
            params=config.get("params"),
            body=config.get("body"),
            auth_config=config.get("auth"),
            rate_limit=config.get("rate_limit"),
            retry_count=config.get("retry_count", 3)
        )
    
    elif source_type == DataSourceType.WEB_SCRAPE:
        return await _extract_web_scrape_impl(
            source_id=source_id,
            url=config.get("url"),
            selectors=config.get("selectors", {}),
            respect_robots=config.get("respect_robots_txt", True),
            rate_limit=config.get("rate_limit", 1),
            max_pages=config.get("max_pages", 1)
        )
    
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

