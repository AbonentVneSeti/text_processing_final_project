import time
import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm


class RateLimiter:
    def __init__(self, max_calls: int, period: float = 1.0):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            while self.calls and now - self.calls[0] > self.period:
                self.calls.popleft()

            if len(self.calls) >= self.max_calls:
                sleep_time = self.calls[0] + self.period - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
                now = time.monotonic()
                while self.calls and now - self.calls[0] > self.period:
                    self.calls.popleft()

            self.calls.append(now)


def create_translator(source, target, rate_limiter):
    rate_limiter.wait()
    return GoogleTranslator(source=source, target=target)


def process_batch(batch, translator_to, translator_back, rate_limiter, max_retries=5):
    trans_en = None
    for attempt in range(max_retries):
        try:
            rate_limiter.wait()
            trans_en = translator_to.translate_batch(batch)
            break
        except Exception as e:
            if 'Too many requests' in str(e) or 'Server Error' in str(e):
                wait = 2 ** attempt
                print(f"Лимит, жду {wait}с (прямой перевод)")
                time.sleep(wait)
            else:
                print(f"Ошибка прямого перевода: {e}")
                return batch, None
    if trans_en is None:
        print("Не удалось выполнить прямой перевод после всех попыток")
        return batch, None

    trans_ru = None
    for attempt in range(max_retries):
        try:
            rate_limiter.wait()
            trans_ru = translator_back.translate_batch(trans_en)
            break
        except Exception as e:
            if 'Too many requests' in str(e) or 'Server Error' in str(e):
                wait = 2 ** attempt
                print(f"Лимит, жду {wait}с (обратный перевод)")
                time.sleep(wait)
            else:
                print(f"Ошибка обратного перевода: {e}")
                return batch, None
    if trans_ru is None:
        print("Не удалось выполнить обратный перевод после всех попыток")
        return batch, None

    return batch, trans_ru


def load_or_create(config_section: dict) -> pd.DataFrame:
    from_lang = config_section.get("from_lang", "ru")
    to_lang = config_section.get("to_lang", "en")
    batch_size = config_section.get("batch_size", 100)
    max_workers = config_section.get("max_workers", 4)
    max_requests_per_second = config_section.get("max_requests_per_second", 4)
    checkpoint_every = config_section.get("checkpoint_every", 100)
    output_file = config_section.get("output_file", "data/backtranslated.parquet")

    if "sentences" in config_section:
        sentences = config_section["sentences"]
    else:
        with open(config_section["input_file"], 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]

    if os.path.exists(output_file):
        existing = pd.read_parquet(output_file)
        done_originals = set(existing['original'].tolist())
        print(f"Найдено {len(existing)} уже обработанных предложений, продолжаем...")
        sentences = [s for s in sentences if s not in done_originals]
    else:
        existing = pd.DataFrame(columns=['original', 'paraphrase'])

    if not sentences:
        print("Все предложения уже обработаны.")
        return existing

    batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]

    rate_limiter = RateLimiter(max_calls=max_requests_per_second, period=1.0)

    translators_to = []
    translators_back = []
    for _ in range(max_workers):
        translators_to.append(create_translator(from_lang, to_lang, rate_limiter))
        translators_back.append(create_translator(to_lang, from_lang, rate_limiter))

    all_originals = []
    all_paraphrases = []
    completed_batches = 0
    processed_futures = set()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, batch in enumerate(batches):
            worker_id = i % max_workers
            fut = executor.submit(process_batch, batch,
                                  translators_to[worker_id],
                                  translators_back[worker_id],
                                  rate_limiter)
            futures.append(fut)

        try:
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Back-translating"):
                processed_futures.add(fut)
                batch, trans_ru = fut.result()
                if trans_ru is not None:
                    all_originals.extend(batch)
                    all_paraphrases.extend(trans_ru)
                completed_batches += 1

                if completed_batches % checkpoint_every == 0 and all_originals:
                    new_df = pd.DataFrame({'original': all_originals[-checkpoint_every*batch_size:],
                                           'paraphrase': all_paraphrases[-checkpoint_every*batch_size:]})
                    updated = pd.concat([existing, new_df], ignore_index=True)
                    updated.to_parquet(output_file, index=False)
                    existing = updated
        except KeyboardInterrupt:
            print("\nПрерывание! Сохраняем незавершённые результаты...")

            for fut in futures:
                if fut.done() and fut not in processed_futures:
                    try:
                        batch, trans_ru = fut.result()
                        if trans_ru is not None:
                            all_originals.extend(batch)
                            all_paraphrases.extend(trans_ru)
                    except Exception:
                        pass

            if all_originals:
                saved_batches = (len(all_originals) // (checkpoint_every * batch_size)) * checkpoint_every
                if len(all_originals) > saved_batches * batch_size:
                    start_idx = saved_batches * batch_size
                    new_df = pd.DataFrame({'original': all_originals[start_idx:],
                                           'paraphrase': all_paraphrases[start_idx:]})
                    updated = pd.concat([existing, new_df], ignore_index=True)
                    updated.to_parquet(output_file, index=False)
                    existing = updated
            print(f"Сохранено. Обработано {len(existing)} предложений.")
            raise

    if all_originals:
        saved_batches = (len(all_originals) // (checkpoint_every * batch_size)) * checkpoint_every
        if len(all_originals) > saved_batches * batch_size:
            start_idx = saved_batches * batch_size
            new_df = pd.DataFrame({'original': all_originals[start_idx:],
                                   'paraphrase': all_paraphrases[start_idx:]})
            updated = pd.concat([existing, new_df], ignore_index=True)
            updated.to_parquet(output_file, index=False)
            existing = updated

    return existing