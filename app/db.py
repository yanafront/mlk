import os
import psycopg2
from pgvector.psycopg2 import register_vector


def get_conn():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    register_vector(conn)
    return conn
