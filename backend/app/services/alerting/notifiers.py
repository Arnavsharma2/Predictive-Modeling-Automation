"""
Notification implementations for alerts.
"""
from typing import List, Dict, Any, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import json

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmailNotifier:
    """Email notification sender."""
    
    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        from_email: Optional[str] = None
    ):
        """
        Initialize email notifier.
        
        Args:
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            smtp_user: SMTP username
            smtp_password: SMTP password
            from_email: From email address
        """
        self.smtp_host = smtp_host or getattr(settings, "SMTP_HOST", None)
        self.smtp_port = smtp_port or getattr(settings, "SMTP_PORT", 587)
        self.smtp_user = smtp_user or getattr(settings, "SMTP_USER", None)
        self.smtp_password = smtp_password or getattr(settings, "SMTP_PASSWORD", None)
        self.from_email = from_email or getattr(settings, "SMTP_FROM_EMAIL", "noreply@analytics-platform.com")
    
    async def send(
        self,
        recipients: List[str],
        subject: str,
        body: str,
        html_body: Optional[str] = None
    ) -> bool:
        """
        Send email notification.
        
        Args:
            recipients: List of email addresses
            subject: Email subject
            body: Plain text body
            html_body: HTML body (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.smtp_host or not self.smtp_user:
            logger.warning("SMTP not configured, skipping email notification")
            return False
        
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.from_email
            msg["To"] = ", ".join(recipients)
            
            # Add plain text part
            text_part = MIMEText(body, "plain")
            msg.attach(text_part)
            
            # Add HTML part if provided
            if html_body:
                html_part = MIMEText(html_body, "html")
                msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email sent to {recipients}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False


class WebhookNotifier:
    """Webhook notification sender."""
    
    async def send(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Send webhook notification.
        
        Args:
            url: Webhook URL
            payload: Payload to send
            headers: Optional headers
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers or {}
                ) as response:
                    if response.status >= 200 and response.status < 300:
                        logger.info(f"Webhook sent to {url}")
                        return True
                    else:
                        logger.error(f"Webhook failed with status {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending webhook: {e}")
            return False


class InAppNotifier:
    """In-app notification (stored in database)."""
    
    async def send(
        self,
        db,
        alert_id: int,
        user_ids: Optional[List[int]] = None
    ) -> bool:
        """
        Store in-app notification.
        
        Args:
            db: Database session
            alert_id: Alert ID
            user_ids: List of user IDs to notify (None = all users)
            
        Returns:
            True if successful, False otherwise
        """
        # In-app notifications are stored as alerts in the database
        # This is a placeholder for future implementation
        # Could create a separate notifications table for users
        return True


# Global instances
email_notifier = EmailNotifier()
webhook_notifier = WebhookNotifier()
in_app_notifier = InAppNotifier()

