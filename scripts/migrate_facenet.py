"""Migration helper: generate FaceNet embeddings for students from an images folder.

Usage:
  python scripts/migrate_facenet.py --folder path/to/images

The script will look for image files whose filenames contain the student's roll number
and will call into `face_utils.update_student_face_encoding` to regenerate and store
encodings (face_recognition and facenet when available).

If no --folder is provided the script will print guidance.
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from app import app, db, Student
from face_utils import batch_encode_faces_from_folder, update_student_face_encoding


def main():
    p = argparse.ArgumentParser(description='Migrate/generate FaceNet encodings for students from image files')
    p.add_argument('--folder', '-f', help='Folder containing student images (filenames containing roll numbers)')
    p.add_argument('--student', '-s', type=int, help='Process single student id instead of folder')
    args = p.parse_args()

    if not args.folder and not args.student:
        print('No action specified. Provide --folder or --student. Example:')
        print('  python scripts/migrate_facenet.py --folder path/to/images')
        return

    with app.app_context():
        if args.folder:
            folder = os.path.abspath(args.folder)
            if not os.path.exists(folder):
                print(f'Folder not found: {folder}')
                return
            print(f'Batch encoding faces from folder: {folder}')
            batch_encode_faces_from_folder(folder)
            print('Batch encoding complete')

        if args.student:
            sid = args.student
            s = Student.query.get(sid)
            if not s:
                print(f'Student id {sid} not found')
                return
            # Attempt to find a matching image in current working directory or prompt
            print(f'Preparing to encode face for student: {s.id} {s.name} ({s.roll_number})')
            # Look for images in current dir matching roll number
            cwd = os.getcwd()
            matches = [f for f in os.listdir(cwd) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and s.roll_number in f]
            if matches:
                img = os.path.join(cwd, matches[0])
                print(f'Found image {img}, encoding...')
                ok = update_student_face_encoding(s.id, img)
                print('OK' if ok else 'Failed')
            else:
                print('No matching image found in current directory. Use --folder to point to images folder.')


if __name__ == '__main__':
    main()
