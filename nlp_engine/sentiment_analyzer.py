import os
import torch
from transformers import pipeline
from pymongo import MongoClient
from tqdm import tqdm


class FinancialSentimentAnalyzer:
    def __init__(self, db_name='post_info'):
        self.client = MongoClient('localhost', 27017)
        self.db = self.client[db_name]

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("[INFO] Device: Apple MPS enabled.")
        else:
            self.device = torch.device("cpu")
            print("[INFO] Device: CPU.")

        self.model_name = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
        self.sentiment_pipe = pipeline("sentiment-analysis", model=self.model_name, device=self.device)

    def _convert_score(self, hf_result):
        label = hf_result['label']
        score = hf_result['score']
        return score if label == 'Positive' else -score

    def analyze_collection(self, collection_name):
        collection = self.db[collection_name]
        query = {"sentiment_score": {"$exists": False}}
        total_docs = collection.count_documents(query)

        if total_docs == 0:
            print(f"[*] No new documents found in '{collection_name}'. Skipping.")
            return

        print(f"[*] Processing '{collection_name}' | Pending records: {total_docs}")
        cursor = collection.find(query)

        for doc in tqdm(cursor, total=total_docs, desc="Inference Progress"):
            title = doc.get('title', '').strip()
            summary = doc.get('summary', '').strip()

            text = f"{title}。{summary}" if summary else title
            text = text.strip('。').strip()

            if not text:
                collection.update_one({"_id": doc["_id"]}, {"$set": {"sentiment_score": 0.0}})
                continue

            try:
                truncated_text = text[:500]
                result = self.sentiment_pipe(truncated_text)[0]
                final_score = self._convert_score(result)

                collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {
                        "sentiment_score": final_score,
                        "sentiment_label": result['label']
                    }}
                )
            except Exception:
                collection.update_one({"_id": doc["_id"]}, {"$set": {"sentiment_score": 0.0}})

    def run_pipeline(self):
        print("-" * 60)
        print("Starting NLP Sentiment Analysis Pipeline")
        print("-" * 60)
        self.analyze_collection(collection_name='news_stcn_上证指数')
        self.analyze_collection(collection_name='post_zssh000001')
        print("[*] Pipeline execution completed.")

if __name__ == "__main__":
    analyzer = FinancialSentimentAnalyzer()
    analyzer.run_pipeline()