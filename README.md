# Face Recognition Auto Attendance System

A comprehensive face recognition-based attendance system with dual WiFi connectivity for classroom management.

## Features

- **Teacher Authentication**: Secure login system for teachers
- **Subject Management**: Manage multiple subjects (3-5 subjects supported)
- **Student Management**: Add and manage students for each subject
- **Face Recognition**: Automatic attendance marking using face recognition
- **Dual WiFi System**: Connect to classroom WiFi for dual system operation
- **Real-time Attendance**: Start/stop attendance sessions with live monitoring
- **Absence Tracking**: Generate absence lists and reports
- **Modern UI**: Beautiful, responsive web interface

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional):
   Create a `.env` file in the project root:
   ```
   SECRET_KEY=your-secret-key-here
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Access the system**:
   - Open your web browser
   - Navigate to `http://localhost:5000`
   - Default login: username=`admin`, password=`admin123`

## Usage

### 1. Teacher Login
- Register a new teacher account or use the default admin account
- Secure authentication with password hashing

### 2. Subject Management
- Add subjects with names and codes
- Each teacher can manage their own subjects
- Support for 3-5 subjects as requested

### 3. Student Management
- Add students to each subject
- Upload face photos for face recognition
- Manage student roll numbers and details

### 4. Attendance Taking
- Start attendance session with face recognition
- Real-time video feed with face detection
- Automatic marking of present students
- Manual override for corrections
- Stop attendance session when complete

### 5. Absence Management
- View absence list for each subject
- Export absence reports
- Send notifications to absent students
- Track attendance history

## System Requirements

- Python 3.7+
- Webcam for face recognition
- WiFi connectivity for dual system operation
- Modern web browser

## Technical Details

- **Backend**: Flask (Python)
- **Database**: SQLite (easily upgradeable to PostgreSQL/MySQL)
- **Face Recognition**: OpenCV + face_recognition library
- **Frontend**: Bootstrap 5 + JavaScript
- **Authentication**: Flask-Login with password hashing

## WiFi Dual System

The system is designed to work with dual WiFi connectivity:
- Primary system connects to classroom WiFi
- Secondary system can connect to backup network
- Automatic failover between systems
- Real-time status monitoring

## Face Recognition Process

1. Students position themselves in front of camera
2. System detects faces in real-time
3. Compares detected faces with registered student faces
4. Automatically marks recognized students as present
5. Updates attendance database in real-time

## Security Features

- Password hashing for teacher accounts
- Session management with Flask-Login
- Secure face data storage
- Input validation and sanitization

## Troubleshooting

### Common Issues:

1. **Camera not working**:
   - Check camera permissions
   - Ensure camera is not used by other applications
   - Try different camera index in code

2. **Face recognition not working**:
   - Ensure students have uploaded face photos
   - Check lighting conditions
   - Verify face_recognition library installation

3. **WiFi connectivity issues**:
   - Check network configuration
   - Verify dual system setup
   - Monitor connection status in dashboard

## Support

For technical support or feature requests, please contact the development team.

## License

This project is developed for educational purposes. Please ensure compliance with privacy laws when handling student data.
