import os

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

client: Client | None = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    try:
        client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    except Exception as exc:
        print(f"Supabase client init failed: {exc}", flush=True)
        client = None
else:
    print(
        "Supabase storage disabled: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY missing.",
        flush=True,
    )
