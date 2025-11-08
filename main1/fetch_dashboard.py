import requests
s = requests.Session()
login_url = 'http://127.0.0.1:5000/login'
with open('fetch_debug.txt','w', encoding='utf-8') as log:
    r = s.get(login_url)
    log.write(f'GET /login status: {r.status_code}\n')
    payload={'username':'admin','password':'admin123'}
    r2 = s.post(login_url, data=payload, allow_redirects=True)
    log.write(f'POST /login status: {r2.status_code} url: {r2.url}\n')
    r3 = s.get('http://127.0.0.1:5000/dashboard')
    log.write(f'GET /dashboard status: {r3.status_code}\n')
    open('dashboard_page.html','w', encoding='utf-8').write(r3.text)
    log.write('Saved dashboard_page.html len=' + str(len(r3.text)) + '\n')
print('done')
