#!/usr/bin/env python3
"""
Create Prefect database if it doesn't exist.
"""
import sys
import asyncio
import asyncpg

async def create_database():
    """Create prefect_db database if it doesn't exist."""
    # Connect to default postgres database
    conn = await asyncpg.connect(
        host='db',
        port=5432,
        user='postgres',
        password='postgres',
        database='postgres'
    )
    
    try:
        # Check if database exists
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            'prefect_db'
        )
        
        if not exists:
            # Create database
            await conn.execute('CREATE DATABASE prefect_db')
            print("Created database 'prefect_db'")
        else:
            print("Database 'prefect_db' already exists")
    finally:
        await conn.close()

if __name__ == '__main__':
    try:
        asyncio.run(create_database())
        sys.exit(0)
    except Exception as e:
        print(f"Error creating database: {e}", file=sys.stderr)
        sys.exit(1)

