import psycopg

from dotenv import load_dotenv
from os import environ as env

load_dotenv()

client = psycopg.connect(env.get("DATABASE_URL"), autocommit=True) if env.get("DATABASE_URL") else None