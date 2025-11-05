"""
Migration helper: copy existing `subject_id` values from `student` table into the
new `student_subject` association table.

Run this once after pulling the many-to-many change:

    py -3 scripts\migrate_student_subjects.py

It will:
- detect whether `subject_id` column exists on the `student` table
- for each student with a non-null subject_id, insert a row into student_subject

Note: this script assumes the updated models are present in `app.py`.
"""
from app import app, db
from app import Student, Subject
from sqlalchemy import text

with app.app_context():
    # Check if subject_id column exists in the students table
    res = db.session.execute(text("PRAGMA table_info(student);"))
    columns = [row[1] for row in res.fetchall()]
    if 'subject_id' not in columns:
        print('No subject_id column found on student table; nothing to migrate.')
    else:
        print('subject_id column detected. Migrating values to student_subject table...')
        # Query all students with a subject_id
        rows = db.session.execute(text('SELECT id, subject_id FROM student WHERE subject_id IS NOT NULL;'))
        migrated = 0
        for sid, subject_id in rows.fetchall():
            # ensure subject exists
            subject = Subject.query.get(subject_id)
            if subject:
                # insert association if not exists
                exists = db.session.execute(
                    text('SELECT 1 FROM student_subject WHERE student_id=:sid AND subject_id=:s'),
                    {'sid': sid, 's': subject_id}
                ).fetchone()
                if not exists:
                    db.session.execute(
                        text('INSERT INTO student_subject(student_id, subject_id) VALUES(:sid, :s)'),
                        {'sid': sid, 's': subject_id}
                    )
                    migrated += 1
        db.session.commit()
        print(f'Migrated {migrated} associations.')
        print('Done. You may remove the old column manually if desired (sqlite has limited ALTER support).')