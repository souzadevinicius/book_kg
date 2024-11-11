import os
from neo4j import GraphDatabase
import logging
from contextlib import contextmanager

# Set logging level for neo4j
logging.getLogger("neo4j").setLevel(logging.WARNING)

class Neo4jConnection:
    def __init__(self):
        self.__uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.__user = os.getenv("NEO4J_USER", "neo4j")
        self.__pwd = os.getenv("NEO4J_PASSWORD", "admin1234")
        self.__driver = None
        self.__connect()

    def __connect(self):
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            logging.error(f"Failed to create the driver: {e}")
            raise

    def close(self):
        if self.__driver:
            self.__driver.close()

    @contextmanager
    def get_session(self, db=None):
        session = self.__driver.session(database=db) if db else self.__driver.session()
        try:
            yield session
        finally:
            session.close()

    def query(self, query, parameters=None, db=None):
        if not self.__driver:
            raise ValueError("Driver not initialized!")

        try:
            with self.get_session(db) as session:
                result = list(session.run(query, parameters))
            return result
        except Exception as e:
            logging.error(f"Query failed: {e}")
            raise

# Usage
try:
    conn = Neo4jConnection()
    # Use conn.query() here
finally:
    conn.close()