import pandas as pd
from pymongo import MongoClient
from pathlib import Path

# 动态获取项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent


class MultimodalFeatureFuser:
    def __init__(self, db_name='post_info'):
        print("=" * 65)
        print("[INFO] Multi-modal Time-series Alignment Engine Started")
        print("[INFO] Aggregating NLP features and merging with Market Data")
        print("=" * 65)

        self.client = MongoClient('localhost', 27017)
        self.db = self.client[db_name]

        # 动态拼接量价原始特征表路径
        input_path = BASE_DIR / "data" / "raw" / "ssec_ultimate_features_2025.csv"
        self.market_df = pd.read_csv(str(input_path))
        self.market_df['date'] = pd.to_datetime(self.market_df['date']).dt.strftime('%Y-%m-%d')

    def aggregate_sentiment(self, collection_name, prefix):
        """按交易日对百万级非结构化数据进行时序降维"""
        print(f"[*] Aggregating sentiment for: {collection_name}")
        collection = self.db[collection_name]

        pipeline = [
            {"$match": {"sentiment_score": {"$exists": True}}},
            {"$project": {
                "date_str": {
                    "$dateToString": {"format": "%Y-%m-%d", "date": "$post_date", "timezone": "Asia/Shanghai"}},
                "sentiment_score": 1,
                "is_positive": {"$cond": [{"$gt": ["$sentiment_score", 0]}, 1, 0]}
            }},
            {"$group": {
                "_id": "$date_str",
                f"{prefix}_avg_sentiment": {"$avg": "$sentiment_score"},
                "total_posts": {"$sum": 1},
                "positive_posts": {"$sum": "$is_positive"}
            }},
            {"$project": {
                "date": "$_id",
                "_id": 0,
                f"{prefix}_avg_sentiment": 1,
                f"{prefix}_post_volume": "$total_posts",
                f"{prefix}_pos_ratio": {"$divide": ["$positive_posts", "$total_posts"]}
            }}
        ]

        cursor = collection.aggregate(pipeline)
        return pd.DataFrame(list(cursor))

    def run_fusion(self):
        guba_df = self.aggregate_sentiment('post_zssh000001', prefix='guba')
        news_df = self.aggregate_sentiment('news_stcn_上证指数', prefix='news')

        final_df = self.market_df.copy()

        if not guba_df.empty:
            final_df = pd.merge(final_df, guba_df, on='date', how='left')
        if not news_df.empty:
            final_df = pd.merge(final_df, news_df, on='date', how='left')

        # 缺失值平滑处理：填充为中性情绪 (0)
        sentiment_cols = [col for col in final_df.columns if 'sentiment' in col or 'ratio' in col or 'volume' in col]
        final_df[sentiment_cols] = final_df[sentiment_cols].fillna(0)

        # 动态拼接并生成目标导出路径
        output_file = BASE_DIR / "data" / "processed" / "LSTM_Multimodal_Dataset_2025.csv"

        # 确保 data/processed 目录物理存在，不存在则创建
        output_file.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(str(output_file), index=False, encoding='utf-8-sig')

        print(f"\n[SUCCESS] 最终数据集对齐完毕！")
        print(f"[INFO] 矩阵维度: {final_df.shape}")
        print(f"[INFO] 文件已保存至: {output_file}")


if __name__ == "__main__":
    fuser = MultimodalFeatureFuser()
    fuser.run_fusion()