from os import environ as env

import redis
from dotenv import load_dotenv
from redis.asyncio import Redis as AsyncRedis
from upstash_redis import Redis as UpstashRedis
from upstash_redis.asyncio import Redis as AsyncUpstashRedis

load_dotenv()

_redis_url = env.get("REDIS_URL")
_upstash_rest_url = env.get("UPSTASH_REDIS_REST_URL")
_upstash_rest_token = env.get("UPSTASH_REDIS_REST_TOKEN")

client = redis.from_url(_redis_url, decode_responses=True) if _redis_url else None
async_client = AsyncRedis.from_url(_redis_url, decode_responses=True) if _redis_url else None

upstash_client = (
    UpstashRedis(url=_upstash_rest_url, token=_upstash_rest_token)
    if _upstash_rest_url and _upstash_rest_token
    else None
)
async_upstash_client = (
    AsyncUpstashRedis(url=_upstash_rest_url, token=_upstash_rest_token)
    if _upstash_rest_url and _upstash_rest_token
    else None
)

if client:
    print("Redis ping:", client.ping())
if upstash_client:
    print("Upstash REST ping:", upstash_client.ping())
