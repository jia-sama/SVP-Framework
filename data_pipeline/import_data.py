import pandas as pd
from pymongo import MongoClient
import os
import math
from pathlib import Path

# 动态获取项目根目录: 当前文件 -> 所在目录(data_pipeline) -> 父目录(项目根目录)
BASE_DIR = Path(__file__).resolve().parent.parent

def import_tables_to_mongodb(db_name='post_info'):
    print("-" * 65)
    print("Database Recovery Pipeline Started")
    print("-" * 65)

    try:
        client = MongoClient('localhost', 27017)
        db = client[db_name]

        # 动态拼接原始数据路径
        tasks = [
            {
                "file": str(BASE_DIR / "data" / "raw" / "EastMoney_post_data.csv"),
                "collection": "post_zssh000001"
            },
            {
                "file": str(BASE_DIR / "data" / "raw" / "news_stcn_上证指数.csv"),
                "collection": "news_stcn_上证指数"
            }
        ]

        for task in tasks:
            file_path = task['file']
            col_name = task['collection']

            if not os.path.exists(file_path):
                print(f"[WARNING] 找不到文件: {file_path}，已跳过。")
                continue

            print(f"[*] 正在读取 {file_path} 并准备导入至集合: {col_name}...")
            df = pd.read_csv(file_path)

            # 强制时间类型转换
            time_columns = ['post_date', 'crawl_time', '日期', '发布时间']
            for col in time_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')

            # 清洗 NaN 记录
            records = []
            for record in df.to_dict(orient='records'):
                clean_record = {}
                for k, v in record.items():
                    if isinstance(v, float) and math.isnan(v): continue
                    if pd.isnull(v): continue
                    clean_record[k] = v
                records.append(clean_record)

            # 幂等性操作：写入前先清空旧表
            if records:
                db[col_name].drop()
                db[col_name].insert_many(records)
                print(f"[SUCCESS] 成功将 {len(records)} 条数据注入 {col_name}！\n")

        print("-" * 65)
        print("所有底层数据导入任务完美收官！")

    except Exception as e:
        print(f"\n[FATAL ERROR] 导入管道发生异常: {e}")


if __name__ == "__main__":
    import_tables_to_mongodb()