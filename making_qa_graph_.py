from neo4j import GraphDatabase
import pandas as pd

# Neo4j connection details (adjust as needed)
NEO4J_URI = "bolt://localhost:7687"  # Default URI for Neo4j running locally
NEO4J_USER = "neo4j"                 # Default username
NEO4J_PASSWORD = "GIft_4438"             # Default password (change if you set a custom one)

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Function to create nodes using Cypher
def create_question_node(tx, question, answer):
    cypher_query = """
    MERGE (n:Question {name: $question, msg_reply: $answer})
    """
    tx.run(cypher_query, question=question, answer=answer)

# Read the CSV file
file_path = r"C:\Users\User\Documents\questions.csv"
data = pd.read_csv(file_path)

# Iterate over the DataFrame and send Cypher commands to Neo4j
with driver.session() as session:
    for index, row in data.iterrows():
        question = row['คำถาม']
        answer = row['คำตอบ']
        session.write_transaction(create_question_node, question, answer)

# Close the Neo4j connection
driver.close()

print("Graph created in Neo4j with nodes for questions and answers.")
