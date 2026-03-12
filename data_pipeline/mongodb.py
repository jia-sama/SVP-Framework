from pymongo import MongoClient


class MongoAPI:
    def __init__(self, db_name, collection_name):
        """
        初始化 MongoDB 连接
        :param db_name: 数据库名称，本项目统一使用 'post_info'
        :param collection_name: 集合名称 ( 'post_zssh000001'  'news_stcn_上证指数')
        """
        try:
            self.client = MongoClient('localhost', 27017, serverSelectionTimeoutMS=5000)
            # 触发连接测试
            self.client.admin.command('ping')
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
        except Exception as e:
            print(f"[DATABASE ERROR] MongoDB 连接失败，请检查数据库是否已启动: {e}")

    def insert_many(self, data_list):
        """
        批量插入数据，并捕获异常
        """
        if not data_list:
            return False

        try:
            self.collection.insert_many(data_list)
            return True
        except Exception as e:
            print(f"[DATABASE ERROR] 批量插入数据失败: {e}")
            return False