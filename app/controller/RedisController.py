import os
import redis
from app.application.utils.Logging import Logger
from app.config.RedisConfig import RedisClusterConnection


logger = Logger("Redis Controller")


class RedisOnlineStore:
    def __init__(self):
        self.params = RedisClusterConnection().getConfig()

    def connect(self):
        # self.master = RedisCluster(startup_nodes=self.params['cluster'],
        #                              password=self.params['password'],
        #                              decode_responses=True)
        self.master = redis.Redis(
            host=self.params["host"], port=self.params["port"], decode_responses=True
        )

        logger.info("Redis Connect success")

    def insertValueRedis(self, name, key, data):
        try:
            self.master.hset(name, key, data)
            logger.info(f"Inserted value success for name:{name} - key:{key}")
        except Exception as ex:
            logger.error(ex)

    def insertMany(self, name, dataset):
        try:
            self.master.hmset(name, dataset)
            return 1
        except Exception as ex:
            return 0

    def checkExists(self, name, key):
        return self.master.hexists(name, key)

    def getDataByKey(self, name, key):
        return self.master.hget(name, key)

    def hgetallByKey(self, key):
        return self.master.hgetall(key)
