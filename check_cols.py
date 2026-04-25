import psycopg2

SUPABASE_URI = "postgresql://postgres.gkksyrejaiqpxchcyvrb:UYLh%2Aq%24b7ayZMR%2B@aws-1-ap-south-1.pooler.supabase.com:5432/postgres"
conn = psycopg2.connect(SUPABASE_URI)
cursor = conn.cursor()
cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'protected_assets'")
print("Table columns:", [r[0] for r in cursor.fetchall()])
cursor.close()
conn.close()