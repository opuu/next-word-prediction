import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import concurrent.futures
import threading
import time

# Lock for thread-safe operations
lock = threading.Lock()

def fetch_and_save_text(url, filename, retries=3, timeout=10):
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    file.write(chunk)

        return True
    except requests.exceptions.RequestException as e:
        if retries > 0:
            print(f"Retrying {url} ({retries} retries left): {e}")
            return fetch_and_save_text(url, filename, retries - 1, timeout)
        else:
            print(f"Failed to fetch {url} after retries: {e}")
            return False

def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def crawl_url(url, visited_urls, urls_to_visit, data_folder, max_depth):
    with lock:
        if url in visited_urls or max_depth <= 0:
            return
        print(f"Visiting {url}")
        visited_urls.add(url)

    filename = os.path.join(data_folder, f"{len(visited_urls)}.txt")
    if fetch_and_save_text(url, filename):
        # Only parse HTML and find new URLs if the file was successfully fetched
        if urlparse(url).path.endswith('.txt'):
            return

        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find and add new internal links
        new_urls = set()
        for link in soup.find_all('a', href=True):
            new_url = urljoin(url, link['href'])
            if is_valid_url(new_url) and urlparse(new_url).netloc == urlparse(url).netloc:
                with lock:
                    if new_url not in visited_urls:
                        new_urls.add(new_url)

        with lock:
            urls_to_visit.update(new_urls)

def main():
    start_urls = [
        "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
        "http://www.gutenberg.org/files/2600/2600-0.txt",
        "http://www.gutenberg.org/files/1342/1342-0.txt",
        "http://www.gutenberg.org/files/2701/2701-0.txt",
        "http://www.gutenberg.org/files/1661/1661-0.txt",
        "http://www.gutenberg.org/files/4300/4300-0.txt",
        "http://www.gutenberg.org/files/11/11-0.txt",
        "http://www.gutenberg.org/files/10/10-0.txt",
        "http://www.gutenberg.org/files/1400/1400-0.txt",
        "http://www.gutenberg.org/files/84/84-0.txt"
    ]
    visited_urls = set()
    urls_to_visit = set(start_urls)
    data_folder = 'data'
    os.makedirs(data_folder, exist_ok=True)

    max_pages_to_visit = 100   # Maximum number of pages to visit
    max_depth = 3              # Maximum depth of internal links to follow
    timeout_seconds = 36000000 # Maximum time to run the crawler (in seconds)
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        while urls_to_visit and len(visited_urls) < max_pages_to_visit and (time.time() - start_time) < timeout_seconds:
            url = urls_to_visit.pop()
            futures.append(executor.submit(crawl_url, url, visited_urls, urls_to_visit, data_folder, max_depth))

        # Wait for all threads to complete
        concurrent.futures.wait(futures)

    print("Crawling finished.")
    print(f"Visited {len(visited_urls)} pages.")

if __name__ == "__main__":
    main()
