#!/usr/bin/env python3
"""
Wait for PostgreSQL to be ready and create MLflow database if it doesn't exist.
"""
import sys
import time
import psycopg2

def wait_for_db(max_retries=30):
    """Wait for PostgreSQL database to be ready."""
    for i in range(max_retries):
        try:
            conn = psycopg2.connect(
                host='db',
                user='postgres',
                password='postgres',
                dbname='postgres'
            )
            conn.close()
            return True
        except psycopg2.OperationalError:
            if i < max_retries - 1:
                time.sleep(1)
            else:
                return False
    return False

def create_database():
    """Create mlflow_db database if it doesn't exist."""
    try:
        # Connect to default postgres database
        conn = psycopg2.connect(
            host='db',
            user='postgres',
            password='postgres',
            dbname='postgres'
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = 'mlflow_db'"
        )
        exists = cursor.fetchone()
        
        if not exists:
            # Create database
            cursor.execute('CREATE DATABASE mlflow_db')
            print("Created database 'mlflow_db'")
        else:
            print("Database 'mlflow_db' already exists")
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating database: {e}", file=sys.stderr)
        return False

if __name__ == '__main__':
    if not wait_for_db():
        print("Failed to connect to database after 30 retries", file=sys.stderr)
        sys.exit(1)
    
    if not create_database():
        sys.exit(1)
    
    sys.exit(0)

