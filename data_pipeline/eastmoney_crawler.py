import time
import random
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from mongodb import MongoAPI


class EastMoneyCrawler:
    def __init__(self, symbol="zssh000001"):
        """
        初始化东方财富股吧爬虫
        :param symbol: 股票代码 (默认上证指数 zssh000001)
        """
        self.symbol = symbol
        self.base_url = f"http://guba.eastmoney.com/list,{self.symbol}_{{}}.html"

        # 伪装请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': f'http://guba.eastmoney.com/list,{self.symbol}.html'
        }

        # 初始化数据库连接 (对齐 NLP 脚本所需的集合名)
        self.db = MongoAPI('post_info', f'post_{self.symbol}')

    def crawl_pages(self, start_page=1, end_page=100):
        print(f"[*] 开始抓取东方财富股吧: {self.symbol} | 页码: {start_page} 到 {end_page}")

        for page in range(start_page, end_page + 1):
            url = self.base_url.format(page)
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.encoding = 'utf-8'  # 防止中文乱码
                soup = BeautifulSoup(response.text, 'html.parser')

                # 定位帖子列表
                post_list = soup.select('.articleh.normal_post')
                page_data = []

                for post in post_list:
                    try:
                        # 提取阅读量和评论数 (后续可用作情绪加权因子)
                        read_count = post.select_one('.l1.a1').text.strip()
                        comment_count = post.select_one('.l2.a2').text.strip()

                        # 提取标题和链接
                        title_tag = post.select_one('.l3.a3 a')
                        title = title_tag.get('title', '')
                        link = title_tag.get('href', '')
                        if link.startswith('/'):
                            link = 'http://guba.eastmoney.com' + link

                        # 提取作者
                        author = post.select_one('.l4.a4').text.strip()

                        # 提取发帖时间 (补全当前年份)
                        post_time_str = post.select_one('.l5.a5').text.strip()
                        current_year = datetime.now().year
                        full_time_str = f"{current_year}-{post_time_str}"
                        post_date = datetime.strptime(full_time_str, "%Y-%m-%d %H:%M")

                        post_item = {
                            'title': title,
                            'read_count': read_count,
                            'comment_count': comment_count,
                            'author': author,
                            'post_date': post_date,
                            'post_url': link,
                            'crawl_time': datetime.now()
                        }
                        page_data.append(post_item)

                    except Exception as e:
                        continue

                # 存入数据库
                if page_data:
                    self.db.insert_many(page_data)
                    print(f"[INFO] 成功抓取第 {page} 页，获取 {len(page_data)} 条散户帖子。")

                # 休眠防封
                time.sleep(random.uniform(1.5, 3.5))

            except Exception as e:
                print(f"[ERROR] 第 {page} 页抓取失败: {e}")
                time.sleep(5)  # 遇错休眠更长时间

        print(f"\n[SUCCESS] 东方财富股吧 {self.symbol} 抓取任务完成！")