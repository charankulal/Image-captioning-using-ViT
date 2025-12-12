# Image Captioning Web Application

A Flask-based web application with user authentication and AI-powered image captioning using Vision Transformer.

## Features

- **User Authentication**: Secure sign-up and sign-in functionality
- **Image Upload**: Drag-and-drop or browse to upload images
- **AI Captioning**: Generate natural language descriptions using Vision Transformer
- **Caption History**: View all previously generated captions
- **Responsive Design**: Works on desktop and mobile devices

## Installation

1. Install Flask dependencies:
```bash
pip install -r flask_app/requirements.txt
```

2. Make sure you have the main project dependencies installed (see root requirements.txt)

## Running the Application

1. Start the Flask server:
```bash
python flask_app/app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. The database will be automatically created on first run

## Usage

1. **Sign Up**: Create a new account with username, email, and password
2. **Sign In**: Log in with your credentials
3. **Upload Image**: Drag and drop an image or click to browse
4. **Generate Caption**: Click "Generate Caption" to get AI-generated description
5. **View History**: Check the history page to see all your past captions

## Configuration

Edit `app.py` to modify:
- `SECRET_KEY`: Change this for production use
- `MAX_CONTENT_LENGTH`: Maximum file upload size (default: 16MB)
- `ALLOWED_EXTENSIONS`: Allowed image file types

## Database Schema

### Users Table
- `id`: Primary key
- `username`: Unique username
- `email`: Unique email address
- `password_hash`: Hashed password
- `created_at`: Account creation timestamp

### Captions Table
- `id`: Primary key
- `user_id`: Foreign key to users
- `image_path`: Path to uploaded image
- `caption`: Generated caption text
- `confidence`: Model confidence score
- `created_at`: Caption generation timestamp

## Security Notes

- Passwords are hashed using Werkzeug's security functions
- Session-based authentication with secure cookies
- File uploads are validated and sanitized
- CSRF protection through Flask's secret key

## Troubleshooting

### Model Loading Issues
- The captioning model loads on first caption generation (may take a few seconds)
- Ensure `checkpoints/best_model.pth` and `checkpoints/vocab.pth` exist
- Check that PyTorch and transformers are installed

### Database Issues
- Delete `flask_app/database.db` to reset the database
- The database will be recreated on next app start

### Upload Issues
- Check that `flask_app/static/uploads/` directory exists and is writable
- Verify file size is under the configured limit
- Ensure image format is supported (png, jpg, jpeg, gif, bmp)

## Production Deployment

Before deploying to production:
1. Change `SECRET_KEY` to a secure random string
2. Set `debug=False` in `app.run()`
3. Use a production WSGI server (e.g., Gunicorn, uWSGI)
4. Configure proper database (PostgreSQL, MySQL)
5. Set up HTTPS/SSL certificates
6. Configure proper file storage (S3, cloud storage)
