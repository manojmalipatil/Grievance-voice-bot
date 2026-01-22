import asyncio
import aiohttp
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:4b"

PROMPT = """My father does not force me to go for a career his choice. He does not want that his son should follow only his profession. He wants that his son should go according to his own choice, suitability, and capability. But my father desires his son should go for a better future.

My mother is a housewife as well as a beautician. She is a lovely woman. My mother is everything to me. She is the one who understands me best and most closely. My mother is my co-partner in everyday work and different issues.

My mother was the most beautiful woman I have ever seen. She is my mother, who shapes me, to become a nobleman through her great, insightful, and elegant activities. My mother motivates me to learn by consenting to the activities which are important for character building and improvement. My mother creates an environment for me to learn enough in a natural manner.

My grandmother is the cutest person of all. In light of her, everybody has to get up early in the morning. She is fond of making sweet dishes and we love what she does. She is a focused woman and because of her everything needs to be in order.

My brother, who is elder than me, is the tallest. He is a Youtuber and is fond of cooking. He loves to play cricket and is a gadget freak. He doesn’t study much but is very sweet and gentlemanly.

I love my family because they are the jewels of my life. They work hard so that we can get anything we desire makes me love and respect my parents considerably more. We play games every night and discuss various topics to spend some quality time together.Family games

Summarise this essay in 20 words"""

PARALLEL_CALLS = 2


async def send_request(session, idx):
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "stream": False
    }

    start = time.time()

    async with session.post(OLLAMA_URL, json=payload) as resp:
        data = await resp.json()

    end = time.time()

    tokens = data.get("eval_count", 0)
    duration_ns = data.get("eval_duration", 0)  # nanoseconds

    duration_s = duration_ns / 1e9 if duration_ns else (end - start)
    tok_per_sec = tokens / duration_s if duration_s > 0 else 0

    print(
        f"[{idx:02d}] "
        f"time={duration_s:.2f}s | "
        f"tokens={tokens} | "
        f"tok/s={tok_per_sec:.2f}"
    )

    return tokens, duration_s


async def main():
    global_start = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = [
            send_request(session, i + 1)
            for i in range(PARALLEL_CALLS)
        ]
        results = await asyncio.gather(*tasks)

    total_tokens = sum(r[0] for r in results)
    total_time = time.time() - global_start

    print("\n==== SUMMARY ====")
    print("Total requests :", PARALLEL_CALLS)
    print("Total tokens   :", total_tokens)
    print("Total time     :", f"{total_time:.2f}s")
    print("Avg tok/sec    :", f"{total_tokens / total_time:.2f}")


asyncio.run(main())