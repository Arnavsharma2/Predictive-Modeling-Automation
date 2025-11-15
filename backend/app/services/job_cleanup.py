"""
Training job cleanup and recovery service.

Handles interrupted training jobs on application startup and shutdown.
"""
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.models.database.training_jobs import TrainingJob, TrainingJobStatus
from app.core.logging import get_logger
from app.core.websocket import manager as ws_manager

logger = get_logger(__name__)


class JobCleanupService:
    """Service for cleaning up interrupted training jobs."""

    @staticmethod
    async def cleanup_interrupted_jobs(db: AsyncSession) -> int:
        """
        Find and mark interrupted training jobs as failed on startup.

        This handles cases where:
        - Backend was restarted while jobs were running
        - Prefect workers crashed
        - Jobs got stuck for any reason

        Args:
            db: Database session

        Returns:
            Number of jobs cleaned up
        """
        logger.info("Starting cleanup of interrupted training jobs...")

        # Find all jobs that are in RUNNING or PENDING state
        result = await db.execute(
            select(TrainingJob).where(
                TrainingJob.status.in_([TrainingJobStatus.RUNNING, TrainingJobStatus.PENDING])
            )
        )
        interrupted_jobs = result.scalars().all()

        if not interrupted_jobs:
            logger.info("No interrupted jobs found")
            return 0

        cleaned_count = 0
        for job in interrupted_jobs:
            # Check how long the job has been in this state
            # Get the last update time (use updated_at if available, otherwise created_at)
            last_update = job.updated_at or job.created_at

            # Handle timezone-aware datetimes
            if last_update.tzinfo is not None:
                # If the datetime is timezone-aware, make current time aware too
                from datetime import timezone
                current_time = datetime.now(timezone.utc)
            else:
                # If naive, use utcnow
                current_time = datetime.utcnow()

            time_since_update = current_time - last_update

            # Mark as failed if it's been stuck for more than a reasonable time
            # For RUNNING jobs: if no update in last 5 minutes, consider it stuck
            # For PENDING jobs: if no update in last 10 minutes, consider it stuck
            max_stale_time = timedelta(minutes=5 if job.status == TrainingJobStatus.RUNNING else 10)

            if time_since_update > max_stale_time:
                logger.warning(
                    f"Found interrupted job {job.id}: status={job.status}, "
                    f"progress={job.progress}, stale_for={time_since_update}"
                )

                # Update the job status
                await db.execute(
                    update(TrainingJob)
                    .where(TrainingJob.id == job.id)
                    .values(
                        status=TrainingJobStatus.FAILED,
                        error_message=(
                            f"Training job was interrupted and automatically marked as failed. "
                            f"Last status: {job.status} at {job.progress}% progress. "
                            f"The training process was likely interrupted during a system restart or crash."
                        ),
                        completed_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                )

                cleaned_count += 1

                # Broadcast update to any connected WebSocket clients
                try:
                    await ws_manager.broadcast_job_update(
                        str(job.id),
                        {
                            "type": "progress_update",
                            "job_id": job.id,
                            "status": "FAILED",
                            "progress": job.progress,
                            "error_message": "Training interrupted during system restart",
                            "updated_at": datetime.utcnow().isoformat()
                        }
                    )
                except Exception as e:
                    logger.error(f"Failed to broadcast cleanup update for job {job.id}: {e}")

        await db.commit()

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} interrupted training job(s)")
        else:
            logger.info("All active jobs appear to be valid")

        return cleaned_count

    @staticmethod
    async def cleanup_old_jobs(db: AsyncSession, days: int = 30) -> int:
        """
        Archive or clean up very old completed/failed jobs.

        Args:
            db: Database session
            days: Number of days to keep jobs

        Returns:
            Number of jobs archived
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Find old completed/failed jobs
        result = await db.execute(
            select(TrainingJob).where(
                TrainingJob.status.in_([
                    TrainingJobStatus.COMPLETED,
                    TrainingJobStatus.FAILED,
                    TrainingJobStatus.CANCELLED
                ]),
                TrainingJob.completed_at < cutoff_date
            )
        )
        old_jobs = result.scalars().all()

        if not old_jobs:
            return 0

        logger.info(f"Found {len(old_jobs)} old jobs to archive (older than {days} days)")

        # For now, just log them. In production, you might want to:
        # 1. Move them to an archive table
        # 2. Delete them
        # 3. Update metadata to mark as archived

        # Example: Mark as archived in metadata
        for job in old_jobs:
            if not job.metadata:
                job.metadata = {}
            job.metadata["archived"] = True
            job.metadata["archived_at"] = datetime.utcnow().isoformat()

        await db.commit()

        return len(old_jobs)

    @staticmethod
    async def check_stuck_jobs(db: AsyncSession) -> list[TrainingJob]:
        """
        Check for jobs that appear to be stuck (running but not progressing).

        This can be called periodically as a health check.

        Args:
            db: Database session

        Returns:
            List of potentially stuck jobs
        """
        # Find jobs that have been RUNNING for more than 2 hours without completion
        two_hours_ago = datetime.utcnow() - timedelta(hours=2)

        result = await db.execute(
            select(TrainingJob).where(
                TrainingJob.status == TrainingJobStatus.RUNNING,
                TrainingJob.started_at < two_hours_ago
            )
        )
        stuck_jobs = result.scalars().all()

        if stuck_jobs:
            logger.warning(f"Found {len(stuck_jobs)} potentially stuck jobs:")
            for job in stuck_jobs:
                runtime = datetime.utcnow() - job.started_at
                logger.warning(
                    f"  Job {job.id}: progress={job.progress}%, "
                    f"runtime={runtime}, last_update={job.updated_at}"
                )

        return stuck_jobs

    @staticmethod
    async def cancel_job(db: AsyncSession, job_id: int, reason: str) -> bool:
        """
        Cancel a running or pending job.

        Args:
            db: Database session
            job_id: Job ID to cancel
            reason: Reason for cancellation

        Returns:
            True if job was cancelled, False otherwise
        """
        result = await db.execute(
            select(TrainingJob).where(TrainingJob.id == job_id)
        )
        job = result.scalar_one_or_none()

        if not job:
            logger.error(f"Job {job_id} not found")
            return False

        if job.status not in [TrainingJobStatus.RUNNING, TrainingJobStatus.PENDING]:
            logger.warning(f"Job {job_id} is not running/pending (status: {job.status})")
            return False

        # Update job status
        await db.execute(
            update(TrainingJob)
            .where(TrainingJob.id == job_id)
            .values(
                status=TrainingJobStatus.CANCELLED,
                error_message=f"Job cancelled: {reason}",
                completed_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        )
        await db.commit()

        logger.info(f"Cancelled job {job_id}: {reason}")

        # Broadcast update
        try:
            await ws_manager.broadcast_job_update(
                str(job_id),
                {
                    "type": "progress_update",
                    "job_id": job_id,
                    "status": "CANCELLED",
                    "progress": job.progress,
                    "error_message": f"Job cancelled: {reason}",
                    "updated_at": datetime.utcnow().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Failed to broadcast cancellation for job {job_id}: {e}")

        return True
