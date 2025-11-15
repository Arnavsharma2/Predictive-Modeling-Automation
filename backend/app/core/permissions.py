"""
Role-based access control (RBAC) permissions and decorators.
"""
from typing import List, Optional
from fastapi import HTTPException, status
from app.models.database.users import User, UserRole


class PermissionChecker:
    """Check if user has required permissions based on role."""

    def __init__(self, allowed_roles: List[UserRole]):
        """
        Initialize permission checker.

        Args:
            allowed_roles: List of roles that are allowed access
        """
        self.allowed_roles = allowed_roles

    def __call__(self, user: User) -> User:
        """
        Check if user has required permissions.

        Args:
            user: User to check permissions for

        Returns:
            User if authorized

        Raises:
            HTTPException: If user doesn't have required permissions
        """
        if user.role not in self.allowed_roles and not user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {[role.value for role in self.allowed_roles]}"
            )
        return user


# Common permission checkers
require_admin = PermissionChecker([UserRole.ADMIN])
require_data_scientist = PermissionChecker([UserRole.ADMIN, UserRole.DATA_SCIENTIST])
require_viewer = PermissionChecker([UserRole.ADMIN, UserRole.DATA_SCIENTIST, UserRole.VIEWER])


def check_model_access(user: User, model_owner_id: int) -> bool:
    """
    Check if user has access to a specific model.

    Args:
        user: User requesting access
        model_owner_id: ID of the model owner

    Returns:
        True if user has access, False otherwise
    """
    # Admins and superusers have access to all models
    if user.is_superuser or user.role == UserRole.ADMIN:
        return True

    # Users can access their own models
    if user.id == model_owner_id:
        return True

    # Data scientists can view all models but not modify others'
    if user.role == UserRole.DATA_SCIENTIST:
        return True

    return False


def check_data_source_access(user: User, data_source_owner_id: int) -> bool:
    """
    Check if user has access to a specific data source.

    Args:
        user: User requesting access
        data_source_owner_id: ID of the data source owner

    Returns:
        True if user has access, False otherwise
    """
    # Admins and superusers have access to all data sources
    if user.is_superuser or user.role == UserRole.ADMIN:
        return True

    # Users can access their own data sources
    if user.id == data_source_owner_id:
        return True

    return False


def can_modify_resource(user: User, resource_owner_id: int) -> bool:
    """
    Check if user can modify a resource.

    Args:
        user: User requesting modification
        resource_owner_id: ID of the resource owner

    Returns:
        True if user can modify, False otherwise
    """
    # Admins and superusers can modify all resources
    if user.is_superuser or user.role == UserRole.ADMIN:
        return True

    # Users can only modify their own resources
    if user.id == resource_owner_id:
        return True

    return False


def can_delete_resource(user: User, resource_owner_id: int) -> bool:
    """
    Check if user can delete a resource.

    Args:
        user: User requesting deletion
        resource_owner_id: ID of the resource owner

    Returns:
        True if user can delete, False otherwise
    """
    # Same as modify for now, can be customized if needed
    return can_modify_resource(user, resource_owner_id)


def has_resource_access(user: Optional[User], resource_owner_id: Optional[int], shared_with: Optional[list] = None) -> bool:
    """
    Check if user has access to a resource (considering ownership and sharing).

    Args:
        user: User requesting access (None if not authenticated)
        resource_owner_id: ID of the resource owner (can be None for legacy resources)
        shared_with: List of user IDs or team IDs that have access (from shared_with JSON field)

    Returns:
        True if user has access, False otherwise
    """
    # If user is not authenticated, no access
    if user is None:
        return False

    # Safely access user attributes (they should be loaded, but handle edge cases)
    try:
        is_superuser = user.is_superuser
        role = user.role
        user_id = user.id
    except Exception:
        # If attributes are expired or not accessible, deny access to be safe
        return False

    # Admins and superusers have access to all resources
    if is_superuser or role == UserRole.ADMIN:
        return True

    # Users can access their own resources
    if resource_owner_id and user_id == resource_owner_id:
        return True

    # Check if resource is shared with this user
    if shared_with:
        if isinstance(shared_with, list):
            # Check if user ID is in shared_with list
            if user_id in shared_with:
                return True
            # TODO: Check team membership if teams are implemented
            # For now, shared_with only supports user IDs

    return False


def filter_by_user_access(query, model_class, user: Optional[User], owner_column_name: str = "created_by"):
    """
    Filter a SQLAlchemy query to only return resources the user has access to.

    Args:
        query: SQLAlchemy query object
        model_class: The model class (e.g., MLModel, DataSource)
        user: Current user (None if not authenticated)
        owner_column_name: Name of the column that stores the owner ID

    Returns:
        Filtered query
    """
    from sqlalchemy import or_, cast, String, func
    from sqlalchemy.dialects.postgresql import JSONB

    # If user is not authenticated, return empty query (no access)
    if user is None:
        # Return a query that matches nothing
        from sqlalchemy import false
        return query.filter(false())

    # Admins and superusers can see all resources
    if user.is_superuser or user.role == UserRole.ADMIN:
        return query

    # Get the owner column and shared_with column
    owner_column = getattr(model_class, owner_column_name)
    shared_with_column = getattr(model_class, "shared_with", None)

    # Build filter conditions
    conditions = [
        owner_column == user.id  # User owns the resource
    ]

    # If shared_with column exists, check if user is in shared_with list
    if shared_with_column is not None:
        # Use PostgreSQL JSON/JSONB operators
        # Cast JSON to text first, then use LIKE for string matching
        try:
            # Convert user.id to string for JSON matching
            user_id_str = str(user.id)
            # Use SQLAlchemy's text operator for PostgreSQL JSON::text casting
            # This casts JSON to text first, then applies LIKE
            from sqlalchemy import text as sql_text
            conditions.append(
                sql_text(f"CAST({shared_with_column.key} AS TEXT) LIKE :user_id_pattern").bindparams(
                    user_id_pattern=f"%{user_id_str}%"
                )
            )
        except Exception:
            # Fallback: if casting doesn't work, we'll filter in Python
            # This is less efficient but more compatible
            pass

    # Combine conditions with OR
    return query.filter(or_(*conditions))
