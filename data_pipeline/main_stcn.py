from stcn_crawler import STCNCrawler
import os

if __name__ == "__main__":
    SYMBOL = '上证指数'
    SCROLL_TIMES = 20
    PORT = 9401

    print("-" * 60)
    print(f"STCN Crawler started. Keyword: {SYMBOL}")
    print("-" * 60)

    try:
        crawler = STCNCrawler(keyword=SYMBOL, port=PORT)
        crawler.crawl_news(target_scroll_times=SCROLL_TIMES)
        print("\nAll tasks processed successfully.")
    except Exception as e:
        print(f"\nCritical Error: {e}")