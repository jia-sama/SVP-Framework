import time
import random
import re
from datetime import datetime
from DrissionPage import ChromiumPage, ChromiumOptions
from pymongo import MongoClient


class STCNCrawler:
    def __init__(self, keyword="上证指数", port=9401):
        self.keyword = keyword
        self.port = port

        self.co = ChromiumOptions()
        self.co.set_local_port(port)
        self.co.set_argument('--no-first-run')
        self.co.set_argument('--disable-blink-features=AutomationControlled')
        self.co.set_argument('--disable-infobars')
        self.co.set_pref('profile.managed_default_content_settings.images', 2)

        self.page = None
        self.current_year = datetime.now().year

        # 数据库连接
        self.client = MongoClient('localhost', 27017)
        self.db = self.client['post_info']
        self.collection = self.db[f'news_stcn_{self.keyword}']

    def _init_browser(self):
        if self.page:
            try:
                self.page.quit()
            except:
                pass
        self.page = ChromiumPage(self.co)
        self.page.run_js('Object.defineProperty(navigator, "webdriver", {get: () => undefined})')

    def parse_time(self, time_str):
        time_str = time_str.strip()
        if len(time_str) >= 16 and re.match(r'\d{4}-\d{2}-\d{2}', time_str):
            dt = datetime.strptime(time_str[:16], "%Y-%m-%d %H:%M")
            self.current_year = dt.year
            return dt
        match = re.search(r'(\d{2})-(\d{2})\s+(\d{2}:\d{2})', time_str)
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            hm = match.group(3)
            return datetime.strptime(f"{self.current_year}-{month:02d}-{day:02d} {hm}", "%Y-%m-%d %H:%M")
        return datetime.now()

    def crawl_news(self, target_scroll_times=20):
        self._init_browser()
        url = f"https://www.stcn.com/article/search.html?search_type=news&keyword={self.keyword}&uncertainty=1&sorter=time"

        try:
            self.page.get(url, timeout=20)
            time.sleep(3)
            collected_urls = set()
            news_data = []

            for scroll_idx in range(target_scroll_times):
                self._check_aliyun_captcha()
                list_items = self.page.s_eles('css:.list.infinite-list li')

                out_of_range = False
                for li in list_items:
                    try:
                        a_tag = li.ele('css:.tt a')
                        if not a_tag: continue
                        link = a_tag.link
                        if link in collected_urls: continue

                        title = a_tag.text.strip()
                        summary_tag = li.ele('css:.text.ellipsis-2')
                        summary = summary_tag.text.strip() if summary_tag else ""

                        info_spans = li.ele('css:.info').eles('tag:span')
                        if not info_spans: continue

                        pub_time_str = info_spans[-1].text
                        source = info_spans[0].text if len(info_spans) > 1 else "证券时报"
                        pub_time = self.parse_time(pub_time_str)

                        if self.current_year < 2025:
                            print(f"[INFO] 已读取到 {self.current_year} 年数据，触发早停。")
                            out_of_range = True
                            break

                        news_item = {
                            'title': title, 'summary': summary, 'source': source,
                            'post_date': pub_time, 'post_url': link, 'crawl_time': datetime.now()
                        }
                        news_data.append(news_item)
                        collected_urls.add(link)
                    except Exception:
                        continue

                print(f"[INFO] 滚动 {scroll_idx + 1}/{target_scroll_times} | 当前解析年份: {self.current_year}")
                if out_of_range: break

                no_more = self.page.ele('css:.no-more.text-center')
                if no_more and "display: none" not in no_more.attr('style'):
                    break

                self.page.scroll.to_bottom()
                time.sleep(random.uniform(2.0, 4.0))

            if news_data:
                self.collection.insert_many(news_data)
                print(f"[DATABASE] 成功将 {len(news_data)} 条记录存入 MongoDB。")
            return news_data

        finally:
            if self.page: self.page.quit()

    def _check_aliyun_captcha(self):
        captcha_mask = self.page.ele('#aliyunCaptcha-window-popup')
        if captcha_mask and "aliyunCaptcha-hidden" not in captcha_mask.attr('class'):
            print("[WARNING] 触发阿里云盾，请手动在浏览器中滑动滑块。")
            while True:
                time.sleep(3)
                if "aliyunCaptcha-hidden" in self.page.ele('#aliyunCaptcha-window-popup').attr('class'):
                    break