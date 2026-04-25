import psycopg2

print("🔌 Knocking on the Docker container's door...")

try:
    # 1. Open the connection
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        dbname="art_defense",
        user="postgres",
        password="secret"
    )
    cursor = conn.cursor()
    
    print("✅ Door opened! Connected to PostgreSQL.")

    # 2. Enable the AI Vector Extension
    print("🧠 Booting up the pgvector ML engine...")
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    
    print("🚀 SUCCESS! Your database is now 512-Dimensional ready.")
    
    # Close the doors
    cursor.close()
    conn.close()

except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("Did you make sure Docker Desktop is running?")