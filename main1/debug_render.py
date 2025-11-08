from app import app
from types import SimpleNamespace
from datetime import datetime
import traceback

def main():
    try:
        with app.test_request_context():
            fake_user = SimpleNamespace(is_authenticated=True, username='admin', email='admin@example.com', created_at=datetime(2023,1,1))
            fake_subjects = [SimpleNamespace(id=1, name='Database Management', code='BCS301', students=[1,2]), SimpleNamespace(id=2, name='Mathematics', code='BCS302', students=[])]
            rendered = app.jinja_env.get_template('dashboard.html').render(current_user=fake_user, subjects=fake_subjects, subjects_json=[{'id':1,'name':'Database Management','code':'BCS301'},{'id':2,'name':'Mathematics','code':'BCS302'}])
            with open('dashboard_render_test.html','w', encoding='utf-8') as f:
                f.write(rendered)
            with open('render_error.txt','w', encoding='utf-8') as ef:
                ef.write('OK: rendered length=' + str(len(rendered)))
    except Exception as e:
        import traceback as tb
        err = tb.format_exc()
        with open('render_error.txt','w', encoding='utf-8') as ef:
            ef.write(err)

if __name__ == '__main__':
    main()
