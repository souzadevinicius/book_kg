from repositories.neo4j_repo import Neo4JRepo
from db.neo4j import Neo4jConnection

class WordsNeo4JRepo(Neo4JRepo):

    def __init__(self, conn:Neo4jConnection = None) -> None:
        Neo4JRepo.__init__(self, conn)


    def init_collection(self):
        # self._conn.query('CREATE CONSTRAINT words IF NOT EXISTS FOR (w:Word) REQUIRE (w.word, w.book, w.chapter) IS UNIQUE')
        pass


    def list(self, query, parameters = None):
        return self._conn.query(query, parameters)

        
    def add_words(self, rows, book_title: str, batch_size=1000):
        try:
            # Adds author nodes to the Neo4j graph as a batch job.
            query = f'''
                        UNWIND $rows AS row
                        MERGE (w1:Word {{word: row.first}})
                        WITH w1, row
                        MERGE (w2:Word {{word: row.second}})
                        WITH row, w1, w2
                        MERGE (w1)-[rel:LINKED 
                        {{book: '{book_title}',
                        chapter: row.chapter,
                        co_occurrence_importance: row.total_importance,
                        co_occurrence_sentence: row.sentence,
                        co_occurrence_paragraph: row.paragraph
                        }}]-(w2)
                        MERGE (b:Book {{title: '{book_title}'}})
                        MERGE (c:BookChapter {{title: row.chapter}})
                        RETURN *
                    '''
            return self.insert_data(query, rows, batch_size)
        except Exception as e:
            print(e,rows)