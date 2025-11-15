"""
CSV file processor for data ingestion.
"""
import pandas as pd
import io
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)


class CSVProcessor:
    """Processor for CSV files."""
    
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
    SUPPORTED_ENCODINGS = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    def __init__(self):
        self.logger = logger
    
    def validate_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Validate CSV file.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            Validation result with status and message
        """
        try:
            # Check file size
            file_size = len(file_content)
            if file_size > self.MAX_FILE_SIZE:
                return {
                    "valid": False,
                    "message": f"File size ({file_size / 1024 / 1024:.2f} MB) exceeds maximum allowed size (100 MB)"
                }
            
            # Check file extension
            if not filename.lower().endswith('.csv'):
                return {
                    "valid": False,
                    "message": "File must have .csv extension"
                }
            
            # Try to read the file with different encodings
            df = None
            encoding_used = None
            for encoding in self.SUPPORTED_ENCODINGS:
                try:
                    df = pd.read_csv(io.BytesIO(file_content), encoding=encoding, nrows=1)
                    encoding_used = encoding
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            
            if df is None:
                return {
                    "valid": False,
                    "message": "Could not parse CSV file. Unsupported encoding or invalid format."
                }
            
            return {
                "valid": True,
                "message": "File is valid",
                "encoding": encoding_used,
                "file_size": file_size
            }
            
        except Exception as e:
            logger.error(f"Error validating CSV file: {e}")
            return {
                "valid": False,
                "message": f"Error validating file: {str(e)}"
            }
    
    def read_csv(
        self,
        file_content: bytes,
        encoding: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Read CSV file into pandas DataFrame.
        
        Args:
            file_content: File content as bytes
            encoding: File encoding (auto-detected if not provided)
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            pandas DataFrame
        """
        if encoding is None:
            # Try to detect encoding
            for enc in self.SUPPORTED_ENCODINGS:
                try:
                    df = pd.read_csv(io.BytesIO(file_content), encoding=enc, **kwargs)
                    logger.info(f"Successfully read CSV with encoding: {enc}")
                    return df
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            raise ValueError("Could not read CSV file with any supported encoding")
        else:
            return pd.read_csv(io.BytesIO(file_content), encoding=encoding, **kwargs)
    
    def get_preview(
        self,
        file_content: bytes,
        n_rows: int = 10,
        encoding: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get preview of CSV file.
        
        Args:
            file_content: File content as bytes
            n_rows: Number of rows to preview
            encoding: File encoding
            
        Returns:
            Dictionary with preview data and metadata
        """
        try:
            df = self.read_csv(file_content, encoding=encoding, nrows=n_rows)
            
            return {
                "columns": df.columns.tolist(),
                "data": df.to_dict(orient='records'),
                "shape": df.shape,
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
        except Exception as e:
            logger.error(f"Error getting CSV preview: {e}")
            raise
    
    def get_statistics(
        self,
        file_content: bytes,
        encoding: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about CSV file.
        
        Args:
            file_content: File content as bytes
            encoding: File encoding
            
        Returns:
            Dictionary with file statistics
        """
        try:
            df = self.read_csv(file_content, encoding=encoding)
            
            stats = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "columns": df.columns.tolist(),
                "missing_values": df.isnull().sum().to_dict(),
                "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "memory_usage": df.memory_usage(deep=True).sum(),
            }
            
            # Add numeric column statistics
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                stats["numeric_statistics"] = df[numeric_cols].describe().to_dict()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting CSV statistics: {e}")
            raise

