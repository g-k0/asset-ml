import psycopg2

SUPABASE_URI = "postgresql://postgres.gkksyrejaiqpxchcyvrb:UYLh%2Aq%24b7ayZMR%2B@aws-1-ap-south-1.pooler.supabase.com:5432/postgres"
conn = psycopg2.connect(SUPABASE_URI)
cursor = conn.cursor()

# Add missing columns
try:
    cursor.execute("ALTER TABLE protected_assets ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW()")
    print("Added created_at column")
except Exception as e:
    print(f"created_at: {e}")

try:
    cursor.execute("ALTER TABLE protected_assets ADD COLUMN IF NOT EXISTS is_augmented BOOLEAN DEFAULT FALSE")
    print("Added is_augmented column")
except Exception as e:
    print(f"is_augmented: {e}")

conn.commit()
print("Done!")
cursor.close()
conn.close()