"""
Utility functions for generating readable feature names from technical feature names.
"""
import re
from typing import Dict, List, Tuple, Optional


def make_feature_name_readable(technical_name: str) -> str:
    """
    Convert a technical feature name to a more readable format.
    
    Examples:
        "numeric__price" -> "Price"
        "categorical__category_0" -> "Category: Option 0"
        "poly_x0 x1" -> "Price × Size"
        "price_x_size" -> "Price × Size"
        "price_rolling_mean_7" -> "Price (7-day average)"
        "price_log" -> "Price (log)"
        "feature_023092" -> "Feature 023092"
    
    Args:
        technical_name: Technical feature name from preprocessing
        
    Returns:
        Readable feature name
    """
    if not technical_name:
        return technical_name
    
    # Remove common prefixes from sklearn transformers
    name = technical_name
    
    # Remove transformer prefixes (numeric__, categorical__, datetime__, text__)
    name = re.sub(r'^(numeric__|categorical__|datetime__|text__)', '', name)
    
    # Handle polynomial features: "poly_x0 x1" or "poly_x0^2"
    if name.startswith('poly_'):
        name = name[5:]  # Remove "poly_" prefix
        # Replace x0, x1, etc. with more readable format
        # This is a simplified version - in practice, we'd need the original column names
        name = re.sub(r'x(\d+)', r'Feature \1', name)
        name = name.replace(' ', ' × ')
        name = name.replace('^', '^')
        return f"Polynomial: {name}"
    
    # Handle interaction features: "price_x_size" -> "Price × Size"
    if '_x_' in name or '_times_' in name:
        parts = re.split(r'_x_|_times_', name)
        readable_parts = [format_column_name(p) for p in parts]
        return ' × '.join(readable_parts)
    
    # Handle plus features: "price_plus_size" -> "Price + Size"
    if '_plus_' in name:
        parts = name.split('_plus_')
        readable_parts = [format_column_name(p) for p in parts]
        return ' + '.join(readable_parts)
    
    # Handle minus features: "price_minus_size" -> "Price - Size"
    if '_minus_' in name:
        parts = name.split('_minus_')
        readable_parts = [format_column_name(p) for p in parts]
        return ' - '.join(readable_parts)
    
    # Handle division features: "price_div_size" -> "Price ÷ Size"
    if '_div_' in name:
        parts = name.split('_div_')
        readable_parts = [format_column_name(p) for p in parts]
        return ' ÷ '.join(readable_parts)
    
    # Handle rolling statistics: "price_rolling_mean_7" -> "Price (7-day average)"
    rolling_match = re.match(r'(.+)_rolling_(mean|std|min|max|median)_(\d+)', name)
    if rolling_match:
        base_col, stat, window = rolling_match.groups()
        stat_names = {
            'mean': 'average',
            'std': 'std dev',
            'min': 'minimum',
            'max': 'maximum',
            'median': 'median'
        }
        readable_base = format_column_name(base_col)
        return f"{readable_base} ({window}-period {stat_names.get(stat, stat)})"
    
    # Handle lag features: "price_lag_1" -> "Price (lag 1)"
    lag_match = re.match(r'(.+)_lag_(\d+)', name)
    if lag_match:
        base_col, lag = lag_match.groups()
        readable_base = format_column_name(base_col)
        return f"{readable_base} (lag {lag})"
    
    # Handle difference features: "price_diff_1" -> "Price (difference)"
    if name.endswith('_diff_1'):
        base_col = name[:-7]
        readable_base = format_column_name(base_col)
        return f"{readable_base} (difference)"
    
    # Handle percentage change: "price_pct_change" -> "Price (% change)"
    if name.endswith('_pct_change'):
        base_col = name[:-11]
        readable_base = format_column_name(base_col)
        return f"{readable_base} (% change)"
    
    # Handle log transforms: "price_log" -> "Price (log)"
    if name.endswith('_log'):
        base_col = name[:-4]
        readable_base = format_column_name(base_col)
        return f"{readable_base} (log)"
    
    if name.endswith('_log1p'):
        base_col = name[:-6]
        readable_base = format_column_name(base_col)
        return f"{readable_base} (log1p)"
    
    # Handle binned features: "price_binned" -> "Price (binned)"
    if name.endswith('_binned'):
        base_col = name[:-7]
        readable_base = format_column_name(base_col)
        return f"{readable_base} (binned)"
    
    # Handle time features: "date_year" -> "Date: Year"
    time_match = re.match(r'(.+)_(year|month|day|dayofweek|dayofyear|week|quarter|hour|minute|second)', name)
    if time_match:
        base_col, time_part = time_match.groups()
        readable_base = format_column_name(base_col)
        time_names = {
            'year': 'Year',
            'month': 'Month',
            'day': 'Day',
            'dayofweek': 'Day of Week',
            'dayofyear': 'Day of Year',
            'week': 'Week',
            'quarter': 'Quarter',
            'hour': 'Hour',
            'minute': 'Minute',
            'second': 'Second'
        }
        return f"{readable_base}: {time_names.get(time_part, time_part)}"
    
    # Handle cyclical encoding: "date_month_sin" -> "Date: Month (sin)"
    if '_sin' in name or '_cos' in name:
        base_name = name.replace('_sin', '').replace('_cos', '')
        trig_type = 'sin' if '_sin' in name else 'cos'
        readable_base = format_column_name(base_name)
        return f"{readable_base} ({trig_type})"
    
    # Handle one-hot encoded features: "category_0" -> "Category: Option 0"
    # Or "categorical__category_0" (already handled prefix removal above)
    onehot_match = re.match(r'(.+)_(\d+)$', name)
    if onehot_match:
        base_col, index = onehot_match.groups()
        readable_base = format_column_name(base_col)
        return f"{readable_base}: Option {index}"
    
    # Handle missing indicators: "price_is_missing" -> "Price (missing indicator)"
    if name.endswith('_is_missing'):
        base_col = name[:-11]
        readable_base = format_column_name(base_col)
        return f"{readable_base} (missing indicator)"
    
    # Handle aggregation features: "price_mean_by_category" -> "Price (mean by Category)"
    agg_match = re.match(r'(.+)_(mean|std|min|max|count|sum)_by_(.+)', name)
    if agg_match:
        base_col, agg_func, group_by = agg_match.groups()
        readable_base = format_column_name(base_col)
        readable_group = format_column_name(group_by)
        agg_names = {
            'mean': 'average',
            'std': 'std dev',
            'min': 'minimum',
            'max': 'maximum',
            'count': 'count',
            'sum': 'sum'
        }
        return f"{readable_base} ({agg_names.get(agg_func, agg_func)} by {readable_group})"
    
    # Handle cluster features
    if name == 'cluster_id':
        return "Cluster ID"
    if name == 'cluster_distance':
        return "Distance to Cluster Center"
    
    # Default: format the column name nicely
    return format_column_name(name)


def format_column_name(name: str) -> str:
    """
    Format a column name to be more readable.
    
    Examples:
        "price" -> "Price"
        "price_usd" -> "Price USD"
        "customer_id" -> "Customer ID"
        "feature_023092" -> "Feature 023092"
    
    Args:
        name: Column name
        
    Returns:
        Formatted column name
    """
    if not name:
        return name
    
    # Replace underscores with spaces
    name = name.replace('_', ' ')
    
    # Capitalize words
    words = name.split()
    capitalized_words = [word.capitalize() for word in words]
    
    return ' '.join(capitalized_words)


def create_feature_name_mapping(
    feature_names: List[str],
    original_columns: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Create a mapping from technical feature names to readable names.
    Prioritizes original column names when available.
    
    Args:
        feature_names: List of technical feature names
        original_columns: List of original dataset column names (before preprocessing)
        
    Returns:
        Dictionary mapping technical names to readable names
    """
    mapping = {}
    
    # If we have original columns, try to map technical features back to them
    if original_columns:
        # Create a mapping from technical features to original columns
        for technical_name in feature_names:
            readable_name = None
            
            # Try to find the original column name in the technical feature name
            # Remove common prefixes first
            clean_name = re.sub(r'^(numeric__|categorical__|datetime__|text__)', '', technical_name)
            
            # Check if technical name directly matches an original column
            if technical_name in original_columns:
                readable_name = format_column_name(technical_name)
            # Check if clean name matches an original column
            elif clean_name in original_columns:
                readable_name = format_column_name(clean_name)
            # Check if technical name starts with an original column name
            else:
                for orig_col in original_columns:
                    # Handle one-hot encoded features: "category_0" from "category"
                    if clean_name.startswith(orig_col + '_') or clean_name.startswith(orig_col + '__'):
                        # Check if it's a one-hot encoded feature
                        onehot_match = re.match(rf'^{re.escape(orig_col)}_(\d+)$', clean_name)
                        if onehot_match:
                            index = onehot_match.group(1)
                            readable_name = f"{format_column_name(orig_col)}: Option {index}"
                            break
                        # Check if it's a transformed feature (log, etc.)
                        elif clean_name.startswith(orig_col + '_log'):
                            readable_name = f"{format_column_name(orig_col)} (log)"
                            break
                        elif clean_name.startswith(orig_col + '_log1p'):
                            readable_name = f"{format_column_name(orig_col)} (log1p)"
                            break
                        elif clean_name.startswith(orig_col + '_binned'):
                            readable_name = f"{format_column_name(orig_col)} (binned)"
                            break
                        # Check for time features
                        elif any(clean_name.startswith(orig_col + f'_{part}') for part in 
                                ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'week', 'quarter', 'hour', 'minute', 'second']):
                            time_part = clean_name.replace(orig_col + '_', '')
                            time_names = {
                                'year': 'Year', 'month': 'Month', 'day': 'Day',
                                'dayofweek': 'Day of Week', 'dayofyear': 'Day of Year',
                                'week': 'Week', 'quarter': 'Quarter',
                                'hour': 'Hour', 'minute': 'Minute', 'second': 'Second'
                            }
                            readable_name = f"{format_column_name(orig_col)}: {time_names.get(time_part, time_part)}"
                            break
            
            # If we found a mapping using original columns, use it
            if readable_name:
                mapping[technical_name] = readable_name
            else:
                # Fall back to heuristic-based readable name
                mapping[technical_name] = make_feature_name_readable(technical_name)
    else:
        # No original columns available, use heuristic-based mapping
        mapping = {
            technical_name: make_feature_name_readable(technical_name)
            for technical_name in feature_names
        }
    
    return mapping


def create_original_column_mapping(
    feature_names: List[str],
    original_columns: List[str]
) -> Dict[str, Optional[str]]:
    """
    Create a mapping from technical feature names to original column names.
    Returns None for engineered features that don't map to a single original column.
    
    Args:
        feature_names: List of technical feature names
        original_columns: List of original dataset column names
        
    Returns:
        Dictionary mapping technical names to original column names (or None)
    """
    mapping = {}
    
    for technical_name in feature_names:
        original_col = None
        
        # Remove common prefixes
        clean_name = re.sub(r'^(numeric__|categorical__|datetime__|text__)', '', technical_name)
        
        # Direct match
        if technical_name in original_columns:
            original_col = technical_name
        elif clean_name in original_columns:
            original_col = clean_name
        # Check if it's derived from an original column (one-hot, log, etc.)
        else:
            for orig_col in original_columns:
                # One-hot encoded: "category_0" -> "category"
                if re.match(rf'^{re.escape(orig_col)}_(\d+)$', clean_name):
                    original_col = orig_col
                    break
                # Transformed features: "price_log" -> "price"
                elif clean_name.startswith(orig_col + '_') and any(
                    clean_name.startswith(orig_col + suffix) 
                    for suffix in ['_log', '_log1p', '_binned', '_year', '_month', '_day']
                ):
                    original_col = orig_col
                    break
        
        mapping[technical_name] = original_col
    
    return mapping


def get_readable_feature_names(feature_names: List[str]) -> List[str]:
    """
    Get readable versions of feature names.
    
    Args:
        feature_names: List of technical feature names
        
    Returns:
        List of readable feature names
    """
    return [make_feature_name_readable(name) for name in feature_names]

