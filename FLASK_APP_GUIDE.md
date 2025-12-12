# Flask Web Application - Quick Start Guide

A complete web application for image captioning with user authentication, built with Flask and powered by Vision Transformer.

## Features

- **User Management**: Secure sign-up and sign-in with password hashing
- **Image Upload**: Drag-and-drop or file browser upload
- **AI-Powered Captions**: Generate captions using Vision Transformer with beam search
- **Caption History**: View and track all your generated captions
- **Responsive Design**: Modern UI that works on all devices
- **SQLite Database**: Lightweight database for user and caption storage

## Quick Start

### 1. Install Dependencies

The Flask app requires all the main project dependencies plus Flask itself:

```bash
# Install main project dependencies (if not already installed)
pip install -r requirements.txt

# Install Flask-specific dependencies
pip install -r flask_app/requirements.txt
```

Or install everything at once:
```bash
pip install Flask Werkzeug torch torchvision transformers pillow numpy
```

### 2. Run the Application

**Option A: Using the run script (Recommended)**
```bash
python flask_app/run.py
```

**Option B: Direct execution**
```bash
python flask_app/app.py
```

### 3. Access the Application

Open your web browser and navigate to:
```
http://localhost:5000
```

## Using the Application

### First Time Setup

1. **Sign Up**: Create a new account
   - Click "Sign Up" in the navigation
   - Enter username, email, and password (minimum 6 characters)
   - Click "Sign Up"

2. **Sign In**: Log in to your account
   - Enter your username and password
   - Click "Sign In"

### Generating Captions

1. **Upload Image**:
   - Drag and drop an image onto the upload area, OR
   - Click "Choose File" to browse for an image
   - Supported formats: PNG, JPG, JPEG, GIF, BMP
   - Maximum file size: 16MB

2. **Generate Caption**:
   - Review the image preview
   - Click "Generate Caption"
   - Wait a few seconds for the AI to process (first generation may take longer as the model loads)
   - View the generated caption and confidence score

3. **View History**:
   - Click "History" in the navigation
   - Browse all your previously generated captions
   - See confidence scores and timestamps

## Application Structure

```
flask_app/
├── app.py                 # Main Flask application
├── run.py                 # Startup script
├── database.db            # SQLite database (created automatically)
├── requirements.txt       # Python dependencies
├── README.md              # Detailed documentation
├── static/
│   ├── css/
│   │   └── style.css     # Application styling
│   └── uploads/           # Uploaded images (created automatically)
└── templates/
    ├── base.html          # Base template
    ├── signin.html        # Sign in page
    ├── signup.html        # Sign up page
    ├── home.html          # Home page with upload
    └── history.html       # Caption history page
```

## Technical Details

### Database Schema

**Users Table**:
- `id`: Primary key (auto-increment)
- `username`: Unique username
- `email`: Unique email address
- `password_hash`: Bcrypt hashed password
- `created_at`: Account creation timestamp

**Captions Table**:
- `id`: Primary key (auto-increment)
- `user_id`: Foreign key to users
- `image_path`: Filename of uploaded image
- `caption`: Generated caption text
- `confidence`: Model confidence score (0-1)
- `created_at`: Generation timestamp

### Model Integration

The Flask app uses the `ImageCaptioningModelManager` from `src/model_manager.py`:

- **Lazy Loading**: Model loads on first caption generation request
- **Beam Search**: Uses beam search with width=5 for better quality captions
- **Confidence Scores**: Calculates and displays confidence for each caption
- **Device Selection**: Automatically uses GPU if available, otherwise CPU

### Security Features

- **Password Hashing**: Werkzeug's `generate_password_hash()` with default settings
- **Session Management**: Flask sessions with secure secret key
- **File Validation**: Checks file extensions and sizes
- **SQL Injection Protection**: SQLite with parameterized queries
- **Login Required**: Routes protected with `@login_required` decorator

## Configuration

Edit `flask_app/app.py` to customize:

```python
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change for production!
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
```

## Troubleshooting

### Model Not Found Error

**Problem**: `FileNotFoundError: Model checkpoint not found`

**Solution**: Ensure you have trained the model and the checkpoint exists:
```bash
ls models/best_model.pth
ls models/vocab.pth
```

Both files must exist. If not, train the model first using the training notebook.

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'flask'` or similar

**Solution**: Install all dependencies:
```bash
pip install -r requirements.txt
pip install -r flask_app/requirements.txt
```

### Database Errors

**Problem**: Database locked or permission errors

**Solution**: Reset the database:
```bash
rm flask_app/database.db
python flask_app/run.py  # Will recreate database
```

### Port Already in Use

**Problem**: `Address already in use`

**Solution**: Either:
- Stop the process using port 5000
- Change the port in `app.py`: `app.run(port=5001)`

### First Caption Takes Long Time

**Behavior**: First caption generation takes 10-20 seconds

**Explanation**: This is normal! The model loads on first request:
- Downloads ViT weights from HuggingFace (~350MB) if not cached
- Loads model checkpoint (~1.7GB)
- Subsequent captions are much faster (1-2 seconds)

## Performance Tips

1. **GPU Acceleration**: Use a CUDA-capable GPU for faster caption generation
2. **Model Caching**: The model stays loaded in memory after first use
3. **Image Size**: Smaller images upload faster but are resized to 224x224 anyway
4. **Beam Width**: Reduce beam_width in `app.py` for faster but lower quality captions

## Production Deployment

Before deploying to production:

1. **Change Secret Key**:
   ```python
   app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'fallback-key')
   ```

2. **Disable Debug Mode**:
   ```python
   app.run(debug=False)
   ```

3. **Use Production Server**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 flask_app.app:app
   ```

4. **Configure HTTPS**: Use nginx or Apache as reverse proxy

5. **Use Production Database**: Migrate from SQLite to PostgreSQL or MySQL

6. **Configure File Storage**: Use cloud storage (S3, Azure Blob) instead of local files

7. **Set Environment Variables**:
   ```bash
   export FLASK_ENV=production
   export SECRET_KEY=your-secure-random-key
   ```

## API Endpoints

The application exposes the following endpoints:

- `GET /`: Landing page (redirects to signin or home)
- `GET /signin`: Sign in page
- `POST /signin`: Process sign in
- `GET /signup`: Sign up page
- `POST /signup`: Process sign up
- `GET /signout`: Sign out user
- `GET /home`: Home page with upload form (requires login)
- `POST /generate_caption`: Generate caption API (requires login)
- `GET /history`: View caption history (requires login)

## Development

To modify the application:

1. **HTML Templates**: Edit files in `flask_app/templates/`
2. **Styling**: Modify `flask_app/static/css/style.css`
3. **Backend Logic**: Edit `flask_app/app.py`
4. **Model Integration**: Modify `src/model_manager.py`

Changes to templates and CSS are reflected immediately in debug mode.
Backend changes require restarting the Flask server.

## Support

For issues or questions:
1. Check this guide and `flask_app/README.md`
2. Review the main project documentation
3. Check the application logs for error messages

## License

This Flask application is part of the Image Captioning project and follows the same license.
