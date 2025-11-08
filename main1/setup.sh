#!/bin/bash
echo "Face Recognition Auto Attendance System Setup"
echo "============================================"

echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Creating database..."
python3 -c "from app import app, db; app.app_context().push(); db.create_all(); print('Database created successfully!')"

echo ""
echo "Setup complete!"
echo ""
echo "Default login credentials:"
echo "Username: admin"
echo "Password: admin123"
echo ""
echo "To start the application, run:"
echo "python3 app.py"
echo ""
echo "Then open your browser and go to: http://localhost:5000"
echo ""
