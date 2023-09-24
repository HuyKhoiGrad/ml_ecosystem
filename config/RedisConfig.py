import os

class RedisClusterConnection():
    def __init__(self):
        self.config = dict()
        
        port = int(os.getenv('DATALAKE_REDIS_PORT', default="6379"))
        lstHost = os.getenv('DATALAKE_REDIS_HOST', default="localhost")

        arrClusterAddress = []
        arrClusterAddress = lstHost.split(',')
        
        nodes = [tuple(x.split(":")) for x in arrClusterAddress]
        startup_nodes = [{"host":x[0], "port":x[1]} for x in nodes]
        
        self.config.update({'cluster': startup_nodes})
        self.config.update({'password': os.getenv('DATALAKE_REDIS_PASSWORD', default="")})

    def getConfig(self):
        return self.config

    
