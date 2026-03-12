from eastmoney_crawler import EastMoneyCrawler

if __name__ == "__main__":
    # 目标：上证指数股吧
    SYMBOL = 'zssh000001'

    # 设定要爬取的页码范围（1页约80条，100页约8000条数据）
    START_PAGE = 1
    END_PAGE = 135000

    print("-" * 60)
    print("Retail Sentiment Crawler (EastMoney Guba) Started.")
    print("-" * 60)

    crawler = EastMoneyCrawler(symbol=SYMBOL)
    crawler.crawl_pages(start_page=START_PAGE, end_page=END_PAGE)