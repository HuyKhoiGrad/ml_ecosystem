import os,sys


class DatabaseConfig(object):
    def __init__(self):
        self.config = {} 

    @property
    def get_config(self):
        return self.config
    
class YugabyteConfig(DatabaseConfig):
    def __init__(self):
        super().__init__()
        self.config.update({'yugabyte_host': os.getenv('YUGABYTE_DB_HOST', default="localhost")})
        self.config.update({'yugabyte_user': os.getenv('YUGABYTE_DB_USER', default="")})
        self.config.update({'yugabyte_pass': os.getenv('YUGABYTE_DB_PASSWORD', default="")})
        self.config.update({'yugabyte_port': os.getenv('YUGABYTE_DB_PORT', default="5432")})
        self.config.update({'yugabyte_dbname': os.getenv('YUGABYTE_DB_DBNAME', default="")})