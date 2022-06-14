

import sqlalchemy  # Package for accessing SQL databases via Python
from dataclasses import dataclass

@dataclass
class PostgresIO:
    
    credentials: str = "postgresql://postgres:twinemac.pannasal@localhost/postgres"
    table: str = "trail_traffic_v2"
    
    def reset_table(self):
        
        engine = sqlalchemy.create_engine(self.credentials)
        con = engine.connect()

        sql = f'drop table {self.table};'
        con.execute(sql)

        sql =f'''CREATE TABLE {self.table}(
        time timestamp,
        day CHAR(20),
        temperature real,
        humidity real,
        conditions CHAR(40),
        object CHAR(20),
        probability JSON,
        frame_url CHAR(70),
        clip_url CHAR(70))'''

        con.execute(sql)
        con.close()

    def update_table(self, timestamp, day, temperature, humidity, conditions, who, probabilities, frame_url, clip_url):
        
        engine = sqlalchemy.create_engine(self.credentials)
        con = engine.connect()

        probabilities = f'''{str(probabilities).replace("'", '"')}'''
        sql = f"INSERT INTO {self.table} values( (TIMESTAMP '{timestamp}'),'{day}', {temperature}, {humidity}, '{conditions}', '{who}', '{probabilities}', '{frame_url}', '{clip_url}')"

        con.execute(sql)
        con.close()

