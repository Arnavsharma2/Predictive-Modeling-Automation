"""
Cloud storage utilities for S3, Azure Blob Storage, and local filesystem.
"""
from typing import Optional, BinaryIO
from io import BytesIO
from pathlib import Path
import asyncio
import boto3
from botocore.exceptions import ClientError
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import AzureError
import aiofiles
import aiofiles.os

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class CloudStorage:
    """Cloud storage client supporting S3, Azure Blob Storage, and local filesystem."""

    def __init__(self):
        self.provider = settings.CLOUD_STORAGE_PROVIDER.lower()

        if self.provider == "s3":
            self._init_s3()
        elif self.provider == "azure":
            self._init_azure()
        elif self.provider == "local":
            self._init_local()
        else:
            raise ValueError(f"Unsupported cloud storage provider: {self.provider}")
    
    def _init_s3(self):
        """Initialize AWS S3 client."""
        if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
            logger.warning("AWS credentials not configured. S3 operations will fail.")
            self.s3_client = None
        else:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION,
            )
        self.bucket_name = settings.S3_BUCKET_NAME
    
    def _init_azure(self):
        """Initialize Azure Blob Storage client."""
        if not settings.AZURE_STORAGE_ACCOUNT_NAME or not settings.AZURE_STORAGE_ACCOUNT_KEY:
            logger.warning("Azure credentials not configured. Blob operations will fail.")
            self.blob_service_client = None
        else:
            connection_string = (
                f"DefaultEndpointsProtocol=https;"
                f"AccountName={settings.AZURE_STORAGE_ACCOUNT_NAME};"
                f"AccountKey={settings.AZURE_STORAGE_ACCOUNT_KEY};"
                f"EndpointSuffix=core.windows.net"
            )
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_name = settings.AZURE_STORAGE_CONTAINER_NAME

    def _init_local(self):
        """Initialize local filesystem storage."""
        self.local_storage_path = Path(settings.LOCAL_STORAGE_PATH)
        self.local_storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Local storage initialized at: {self.local_storage_path}")
    
    async def upload_file(
        self,
        file_content: bytes,
        file_path: str,
        content_type: Optional[str] = None
    ) -> str:
        """
        Upload a file to cloud storage.

        Args:
            file_content: File content as bytes
            file_path: Path/key for the file in storage
            content_type: MIME type of the file

        Returns:
            URL or path to the uploaded file
        """
        if self.provider == "s3":
            return await self._upload_to_s3(file_content, file_path, content_type)
        elif self.provider == "azure":
            return await self._upload_to_azure(file_content, file_path, content_type)
        elif self.provider == "local":
            return await self._upload_to_local(file_content, file_path)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _upload_to_s3(
        self,
        file_content: bytes,
        file_path: str,
        content_type: Optional[str] = None
    ) -> str:
        """Upload file to S3."""
        if not self.s3_client or not self.bucket_name:
            raise ValueError("S3 client not initialized or bucket name not configured")
        
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            # Run synchronous boto3 operation in thread pool
            await asyncio.to_thread(
                self.s3_client.put_object,
                Bucket=self.bucket_name,
                Key=file_path,
                Body=file_content,
                **extra_args
            )
            
            url = f"s3://{self.bucket_name}/{file_path}"
            logger.info(f"File uploaded to S3: {url}")
            return url
        except ClientError as e:
            logger.error(f"Error uploading to S3: {e}")
            raise
    
    async def _upload_to_azure(
        self,
        file_content: bytes,
        file_path: str,
        content_type: Optional[str] = None
    ) -> str:
        """Upload file to Azure Blob Storage."""
        if not self.blob_service_client or not self.container_name:
            raise ValueError("Azure client not initialized or container name not configured")

        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=file_path
            )

            # Run synchronous Azure operation in thread pool
            await asyncio.to_thread(
                blob_client.upload_blob,
                file_content,
                overwrite=True,
                content_settings=None if not content_type else {"content_type": content_type}
            )

            url = blob_client.url
            logger.info(f"File uploaded to Azure: {url}")
            return url
        except AzureError as e:
            logger.error(f"Error uploading to Azure: {e}")
            raise

    async def _upload_to_local(
        self,
        file_content: bytes,
        file_path: str
    ) -> str:
        """Upload file to local filesystem."""
        try:
            full_path = self.local_storage_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file asynchronously
            async with aiofiles.open(full_path, 'wb') as f:
                await f.write(file_content)

            logger.info(f"File saved to local storage: {full_path}")
            return str(full_path)
        except Exception as e:
            logger.error(f"Error saving to local storage: {e}")
            raise
    
    async def download_file(self, file_path: str) -> bytes:
        """
        Download a file from cloud storage.

        Args:
            file_path: Path/key of the file in storage

        Returns:
            File content as bytes
        """
        if self.provider == "s3":
            return await self._download_from_s3(file_path)
        elif self.provider == "azure":
            return await self._download_from_azure(file_path)
        elif self.provider == "local":
            return await self._download_from_local(file_path)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _download_from_s3(self, file_path: str) -> bytes:
        """Download file from S3."""
        if not self.s3_client or not self.bucket_name:
            raise ValueError("S3 client not initialized or bucket name not configured")

        try:
            # Extract key from S3 URL if full URL is provided
            key = file_path
            if file_path.startswith("s3://"):
                # Parse s3://bucket/key format
                parts = file_path.replace("s3://", "").split("/", 1)
                if len(parts) == 2:
                    bucket, key = parts
                    # If bucket doesn't match configured bucket, log warning
                    if bucket != self.bucket_name:
                        logger.warning(f"S3 URL bucket '{bucket}' doesn't match configured bucket '{self.bucket_name}', using configured bucket")
                else:
                    # Just a bucket, no key
                    raise ValueError(f"Invalid S3 URL format: {file_path}")

            logger.info(f"Downloading from S3: bucket={self.bucket_name}, key={key}")

            # Run synchronous boto3 operation in thread pool
            response = await asyncio.to_thread(
                self.s3_client.get_object,
                Bucket=self.bucket_name,
                Key=key
            )
            return await asyncio.to_thread(response['Body'].read)
        except ClientError as e:
            logger.error(f"Error downloading from S3 (bucket={self.bucket_name}, key={key}): {e}")
            raise
    
    async def _download_from_azure(self, file_path: str) -> bytes:
        """Download file from Azure Blob Storage."""
        if not self.blob_service_client or not self.container_name:
            raise ValueError("Azure client not initialized or container name not configured")

        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=file_path
            )
            # Run synchronous Azure operation in thread pool
            download_stream = await asyncio.to_thread(blob_client.download_blob)
            return await asyncio.to_thread(download_stream.readall)
        except AzureError as e:
            logger.error(f"Error downloading from Azure: {e}")
            raise

    async def _download_from_local(self, file_path: str) -> bytes:
        """Download file from local filesystem."""
        try:
            # Handle both absolute and relative paths
            if Path(file_path).is_absolute():
                full_path = Path(file_path)
            else:
                full_path = self.local_storage_path / file_path

            async with aiofiles.open(full_path, 'rb') as f:
                content = await f.read()

            logger.info(f"File loaded from local storage: {full_path}")
            return content
        except Exception as e:
            logger.error(f"Error loading from local storage: {e}")
            raise
    
    async def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from cloud storage.

        Args:
            file_path: Path/key of the file to delete

        Returns:
            True if successful, False otherwise
        """
        if self.provider == "s3":
            return await self._delete_from_s3(file_path)
        elif self.provider == "azure":
            return await self._delete_from_azure(file_path)
        elif self.provider == "local":
            return await self._delete_from_local(file_path)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _delete_from_s3(self, file_path: str) -> bool:
        """Delete file from S3."""
        if not self.s3_client or not self.bucket_name:
            raise ValueError("S3 client not initialized or bucket name not configured")
        
        try:
            # Run synchronous boto3 operation in thread pool
            await asyncio.to_thread(
                self.s3_client.delete_object,
                Bucket=self.bucket_name,
                Key=file_path
            )
            logger.info(f"File deleted from S3: {file_path}")
            return True
        except ClientError as e:
            logger.error(f"Error deleting from S3: {e}")
            return False
    
    async def _delete_from_azure(self, file_path: str) -> bool:
        """Delete file from Azure Blob Storage."""
        if not self.blob_service_client or not self.container_name:
            raise ValueError("Azure client not initialized or container name not configured")

        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=file_path
            )
            # Run synchronous Azure operation in thread pool
            await asyncio.to_thread(blob_client.delete_blob)
            logger.info(f"File deleted from Azure: {file_path}")
            return True
        except AzureError as e:
            logger.error(f"Error deleting from Azure: {e}")
            return False

    async def _delete_from_local(self, file_path: str) -> bool:
        """Delete file from local filesystem."""
        try:
            # Handle both absolute and relative paths
            if Path(file_path).is_absolute():
                full_path = Path(file_path)
            else:
                full_path = self.local_storage_path / file_path

            if full_path.exists():
                await aiofiles.os.remove(full_path)
                logger.info(f"File deleted from local storage: {full_path}")
                return True
            else:
                logger.warning(f"File not found in local storage: {full_path}")
                return False
        except Exception as e:
            logger.error(f"Error deleting from local storage: {e}")
            return False


# Global instance
cloud_storage = CloudStorage()

