import psycopg2

SUPABASE_URI = "postgresql://postgres.gkksyrejaiqpxchcyvrb:UYLh%2Aq%24b7ayZMR%2B@aws-1-ap-south-1.pooler.supabase.com:5432/postgres"
conn = psycopg2.connect(SUPABASE_URI)
c = conn.cursor()

try:
    c.execute("CREATE TABLE IF NOT EXISTS asset_metadata (id SERIAL PRIMARY KEY, asset_filename TEXT, metadata_key TEXT, metadata_value TEXT)")
    print("Created asset_metadata table")
except Exception as e:
    print(f"Table: {e}")

conn.commit()
c.close()
conn.close()