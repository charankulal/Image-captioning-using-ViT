# Flask App Quick Start

## The Problem You Encountered

The error **"Model not loaded! Call load() first"** occurred because:
1. You were running Flask with system Python (without PyTorch installed)
2. The correct environment is `.venv` which has all dependencies installed
3. The vocab.pth file needed special handling (now fixed!)

## Solution: Use the Virtual Environment

### Windows Users

**Double-click this file to start the app:**
```
start_flask_app.bat
```

Or from Command Prompt:
```cmd
start_flask_app.bat
```

### Linux/Mac/Git Bash Users

```bash
./start_flask_app.sh
```

Or manually:
```bash
source .venv/Scripts/activate  # or .venv/bin/activate on Linux/Mac
python flask_app/run.py
```

## What Was Fixed

1. **Vocabulary Loading Issue**: The `vocab.pth` was saved from a Jupyter notebook where the `Vocabulary` class was defined in `__main__`. I updated `src/model_manager.py` to inject the class into the `__main__` namespace before loading, which resolves the unpickling error.

2. **Environment Issue**: Created startup scripts (`start_flask_app.bat` and `start_flask_app.sh`) that automatically activate the virtual environment before starting Flask.

## Using the Application

1. **Start the server** using one of the methods above

2. **Open your browser** to `http://localhost:5000`

3. **Sign up** for a new account:
   - Username: any unique username
   - Email: any valid email format
   - Password: minimum 6 characters

4. **Sign in** with your credentials

5. **Upload an image**:
   - Drag and drop an image onto the upload area, OR
   - Click "Choose File" to browse
   - Supported: PNG, JPG, JPEG (max 16MB)

6. **Generate caption**:
   - Click "Generate Caption"
   - Wait 10-20 seconds on first use (model loading)
   - Subsequent captions take only 1-2 seconds

7. **View history**: Click "History" to see all your past captions

## Troubleshooting

### Still Getting Import Errors?

Make sure you're using the venv Python:
```bash
# Check which Python you're using
which python  # Should show .venv/Scripts/python

# If not, activate venv first
.venv\Scripts\activate.bat  # Windows
source .venv/Scripts/activate  # Linux/Mac
```

### Model Taking Too Long?

First caption generation takes 10-20 seconds because:
- Loading ViT weights (~350MB) from HuggingFace
- Loading model checkpoint (~1.7GB)
- This only happens once! Subsequent captions are fast

### Port 5000 Already in Use?

Change the port in `flask_app/run.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change 5000 to 5001
```

### Database Errors?

Reset the database:
```bash
rm flask_app/database.db
python flask_app/run.py  # Will recreate
```

## Technical Details

- **Model**: Vision Transformer (ViT) encoder + Transformer decoder
- **Beam Search**: Uses beam width=5 for better caption quality
- **GPU**: Automatically uses CUDA if available
- **Database**: SQLite for user and caption storage
- **Sessions**: Secure session-based authentication

## File Structure

```
flask_app/
├── app.py              # Main Flask application
├── run.py              # Startup script
├── database.db         # SQLite database (auto-created)
├── templates/          # HTML templates
├── static/
│   ├── css/           # Styling
│   └── uploads/       # User images (auto-created)
├── README.md          # Detailed documentation
└── requirements.txt   # Dependencies

src/
├── model.py           # Model architecture
├── vocabulary.py      # Vocabulary class
└── model_manager.py   # Model API (FIXED!)

models/
├── best_model.pth     # Trained model checkpoint
└── vocab.pth          # Vocabulary (FIXED to load properly!)
```

## What's Next?

Now that the Flask app is working, you can:
- Generate captions for your images through the web interface
- Build upon this to create a full-featured image captioning service
- Deploy to a production server (see FLASK_APP_GUIDE.md for details)
- Integrate the model into other applications using the model_manager API

Enjoy your image captioning web application! 🎉
