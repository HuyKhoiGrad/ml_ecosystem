import pandas as pd 
import psycopg2

from application.utils.Logging import Logger


logger = Logger('YugaByteDB Controller')
class YugaByteDBController():
    def __init__(self, params):
        super().__init__()   
        self.params = params

    def connect(self):
        self.conn_dwh = psycopg2.connect(
            user=self.params['yugabyte_user'],
            password=self.params['yugabyte_pass'],
            host=self.params['yugabyte_host'],
            port=self.params['yugabyte_port'],
            dbname=self.params['yugabyte_dbname'],
            options="-c search_path=energy,public"
            # load_balance = 'true'
        )
        self.conn_dwh.set_session(autocommit=True)
        self.cursor = self.conn_dwh.cursor()
        logger.info("YugaByteDB connected successfully!!!")

    def insert_data(self, table_name: str,
                    update_data: pd.DataFrame,
                    constraint_key: str):   
        list_records = update_data.to_dict("records")
        for data in list_records:
            try:
                v_sql = 'insert into %s (' %table_name+ ', '.join(data.keys()) + ') values( %s' + \
                        ', %s' * (len(data.keys()) - 1) + ')'
                v_sql = v_sql + " on conflict on constraint %s do update set " %constraint_key
                v_fields = [f + ' = EXCLUDED.' + f for f in list(data.keys())]
                v_stmt_update = ', '.join(v_fields)
                v_sql = v_sql + v_stmt_update
                data_values = list(data.values())           
                data_values = list(map(lambda x: x if x != '' else None, data_values))
                self.cursor.execute(v_sql, data_values)
                # print(v_sql%tuple(data.values()))
                print('Data Insert: {}', data_values)
            except Exception as ex:
                print(str(ex))
                logger.info("[SQL: {} - With Data: {} - Error: {}]".format(v_sql, list(data.values()), str(ex)))

    def get_data(self, sql, col_name):
        self.cursor.execute(sql)
        data = self.cursor.fetchall()
        pd_data = pd.DataFrame(data, columns = col_name)
        return pd_data
    
    def get_all_table(self, schema):
        self.cursor.execute(f"""SELECT table_name FROM information_schema.tables
                        WHERE table_schema = '{schema}'""")
        for table in self.cursor.fetchall():
            print(table)
    
    def query(self, sql): 
        self.cursor.execute(sql)
        data = self.cursor.fetchall()
        return data
    
    def close(self):
        self.conn_dwh.close()
        logger.info("YugaByteDB close successfully!!!")

    def __del__(self): 
        self.close()
