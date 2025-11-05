from app import app

def test_login_page():
    with app.test_client() as client:
        response = client.get('/login')
        print(f'Status: {response.status_code}')
        print(f'Content length: {len(response.data)}')
        if response.status_code == 200:
            print('Login page loads successfully')
        else:
            print('Error loading login page')

if __name__ == "__main__":
    test_login_page()
