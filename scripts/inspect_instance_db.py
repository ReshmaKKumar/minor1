import sqlite3, os
DB='instance/attendance.db'
print('DB exists:', os.path.exists(DB))
if not os.path.exists(DB):
    raise SystemExit('database not found')
conn=sqlite3.connect(DB)
cur=conn.cursor()
print('\nTABLES:')
for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall():
    print(' -', r[0])
print('\nPRAGMA table_info(subject):')
try:
    for r in cur.execute('PRAGMA table_info(subject);').fetchall():
        print('  ', r)
    print('\nSUBJECT ROWS:')
    for r in cur.execute('SELECT id,name,code,teacher_id,created_at FROM subject').fetchall():
        print('  ', r)
except Exception as e:
    print('error inspecting subject table:', e)

conn.close()