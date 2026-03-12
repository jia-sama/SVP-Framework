import pymongo
from tqdm import tqdm
import os
import jieba
from pathlib import Path

# 动态获取项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

class FinancialSentimentCalibrator:
    def __init__(self, db_name='post_info'):
        print("=" * 65)
        print("[INFO] Cascade Calibration Engine v2.0 Initialized")
        print("[INFO] Architecture: Business Rules -> Yao(2021) Lexicon -> RoBERTa")
        print("=" * 65)

        self.client = pymongo.MongoClient('localhost', 27017)
        self.db = self.client[db_name]

        # 动态挂载外部知识库路径
        formal_path = BASE_DIR / "data" / "lexicons" / "Yao2021_formal_sentiment_score.txt"
        informal_path = BASE_DIR / "data" / "lexicons" / "Yao2021_informal_sentiment_score.txt"

        self.formal_lexicon = self._load_weighted_lexicon(str(formal_path))
        self.informal_lexicon = self._load_weighted_lexicon(str(informal_path))

        # 业务规则库 (兼容散户错别字)
        self.strong_neg_rules = ["做空", "融券", "融卷", "清仓", "跌停", "天台", "销户", "骗局", "退市", "白忙活", "玩了个寂寞"]
        self.strong_pos_rules = ["做多", "满仓", "逻辑硬", "龙头", "打板", "涨停", "起飞", "主升浪", "开门红", "长红", "发财"]
        self.neutral_rules = ["总结", "小结", "复盘", "记录", "反思", "日记", "分析", "体会"]

    def _load_weighted_lexicon(self, filepath):
        lexicon_map = {}
        if not os.path.exists(filepath):
            print(f"[ERROR] Lexicon file not found: {filepath}")
            return lexicon_map
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        lexicon_map[parts[0]] = float(parts[1])
                    except ValueError:
                        continue
        return lexicon_map

    def _calculate_lexicon_polarity(self, text, lexicon_map):
        if not text or not lexicon_map: return 0.0
        return sum(lexicon_map.get(token, 0.0) for token in jieba.lcut(text))

    def calibrate_collection(self, collection_name, text_type):
        collection = self.db[collection_name]
        query = {"sentiment_score": {"$exists": True}}
        total_docs = collection.count_documents(query)

        if total_docs == 0: return

        print(f"\n[*] Processing collection: {collection_name} | Total records: {total_docs}")
        active_lexicon = self.informal_lexicon if text_type == 'informal' else self.formal_lexicon

        cursor = collection.find(query)
        calibrated_count = 0

        for doc in tqdm(cursor, total=total_docs, desc=f"Calibrating {collection_name}"):
            title = doc.get('title', '')
            original_score = doc.get('sentiment_score', 0.0)

            # 保护原始推断现场，为消融实验留痕
            if 'raw_sentiment_score' not in doc:
                collection.update_one({"_id": doc["_id"]}, {"$set": {"raw_sentiment_score": original_score}})

            current_raw = doc.get('raw_sentiment_score', original_score)
            new_score = current_raw
            is_calibrated = False

            lexicon_score = self._calculate_lexicon_polarity(title, active_lexicon)

            # 级联判别网络 (Cascade Discrimination)
            # 第一层：业务硬逻辑绝对压制
            if any(word in title for word in self.strong_neg_rules):
                new_score = -0.95
                is_calibrated = True
            elif any(word in title for word in self.strong_pos_rules):
                new_score = 0.95
                is_calibrated = True

            # 第二层：伪情绪衰减
            elif any(word in title for word in self.neutral_rules):
                if -1.5 < lexicon_score < 1.5:
                    new_score = current_raw * 0.1
                    is_calibrated = True

            # 第三层：学术词典极性兜底
            if not is_calibrated:
                if lexicon_score <= -1.0:
                    new_score = -0.95
                    is_calibrated = True
                elif lexicon_score >= 1.0:
                    new_score = 0.95
                    is_calibrated = True

            if is_calibrated:
                calibrated_count += 1
                new_label = 'Positive' if new_score > 0 else 'Negative' if new_score < 0 else 'Neutral'
                collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {
                        "sentiment_score": new_score,
                        "sentiment_label": new_label,
                        "lexicon_net_score": lexicon_score,
                        "is_calibrated": True
                    }}
                )

        print(f"[INFO] Completed {collection_name}. Calibrated: {calibrated_count} documents.")

    def run(self):
        jieba.setLogLevel(jieba.logging.INFO)
        self.calibrate_collection('post_zssh000001', text_type='informal')
        self.calibrate_collection('news_stcn_上证指数', text_type='formal')
        print("\n[INFO] Cascade Calibration finished successfully.")


if __name__ == "__main__":
    calibrator = FinancialSentimentCalibrator()
    calibrator.run()