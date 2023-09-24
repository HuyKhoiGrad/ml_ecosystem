import os
from rediscluster import RedisCluster

class RedisOnlineStore():
    def __init__(self, params):
        self.params = params

    def connect(self):
        self.master = RedisCluster(startup_nodes=self.params['cluster'], 
                                     password=self.params['password'], 
                                     decode_responses=True)
        
    def insertValueRedis(self, name, key, data):
        try:
            self.master.hset(name, key, data)
        except Exception as ex:
            print(ex)

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

    def __del__(self):
        print('Close Redis connection!')
        self.master.connection.disconnect()