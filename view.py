import sqlite3
from pprint import pprint

DB_PATH = "grievance.db"
GRIEVANCE_ID = "30b8f36b-1e87-413e-ace8-35890ffc74c7"   # 👈 change this to the ID you want

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row   # enables column-name access
cursor = conn.cursor()

cursor.execute("""
    SELECT *
    FROM grievances
    WHERE id = ?
""", (GRIEVANCE_ID,))

row = cursor.fetchone()

if row:
    print("\n" + "=" * 60)
    print(f"GRIEVANCE DETAILS (ID: {GRIEVANCE_ID})")
    print("=" * 60)

    for key in row.keys():
        value = row[key]
        print(f"{key:25} : {value}")

    print("=" * 60 + "\n")
else:
    print(f"❌ No grievance found with ID = {GRIEVANCE_ID}")

conn.close()
