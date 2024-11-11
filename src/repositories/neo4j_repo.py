from typing import List, Dict, Any
from neo4j import GraphDatabase, Session

class Neo4JRepo:
    def __init__(self, conn: 'Neo4jConnection' = None) -> None:
        self._conn = conn or Neo4jConnection(user="neo4j", password="admin1234")

    def insert_data(self, query: str, rows: List[Dict[str, Any]], batch_size: int = 1) -> List[Any]:
        """
        Insert data into Neo4j database in batches.

        Args:
            query (str): Cypher query to execute.
            rows (List[Dict[str, Any]]): List of dictionaries containing data to insert.
            batch_size (int): Number of rows to insert in each batch. Default is 1000.

        Returns:
            List[Any]: List of results from all successful batch insertions.
        """
        results = []
        total_batches = (len(rows) + batch_size - 1) // batch_size

        for batch in range(total_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(rows))
            batch_data = rows[start_idx:end_idx]

            try:
                result = self._conn.query(query, parameters={'rows': batch_data})
                results.extend(result)
            except Exception as e:
                print(f"Error in batch {batch + 1}/{total_batches}: {str(e)}")
                # Optionally, implement retry logic or logging here

        return results

class Neo4jConnection:
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def query(self, query: str, parameters: Dict[str, Any] = None) -> List[Any]:
        with self._driver.session() as session:
            result = session.run(query, parameters)
            return list(result)

    def close(self) -> None:
        self._driver.close()