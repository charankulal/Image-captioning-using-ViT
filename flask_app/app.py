import os
import sys
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
import sqlite3
from datetime import datetime

# Add parent directory to path to import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model_manager import ImageCaptioningModelManager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['DATABASE'] = 'flask_app/database.db'
app.config['UPLOAD_FOLDER'] = 'flask_app/static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('flask_app/static/css', exist_ok=True)

# Initialize the image captioning model
caption_model = None

def get_caption_model():
    """Lazy load the captioning model"""
    global caption_model
    if caption_model is None:
        try:
            # Use models directory which contains the actual checkpoint
            checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            print(f"[DEBUG] Loading model from: {checkpoint_dir}")
            print(f"[DEBUG] Files in checkpoint dir: {os.listdir(checkpoint_dir) if os.path.exists(checkpoint_dir) else 'Directory not found'}")

            caption_model = ImageCaptioningModelManager(checkpoint_dir=checkpoint_dir)
            caption_model.load()
            print("[DEBUG] Model loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load caption model: {str(e)}")
    return caption_model

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_db():
    """Get database connection"""
    db = sqlite3.connect(app.config['DATABASE'])
    db.row_factory = sqlite3.Row
    return db

def init_db():
    """Initialize the database"""
    db = get_db()
    db.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS captions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            caption TEXT NOT NULL,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    db.commit()
    db.close()

def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('signin'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    """Landing page - redirect to home if logged in, else to signin"""
    if 'user_id' in session:
        return redirect(url_for('home'))
    return redirect(url_for('signin'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User signup page"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Validation
        if not username or not email or not password:
            flash('All fields are required.', 'error')
            return render_template('signup.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('signup.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('signup.html')

        # Check if user exists
        db = get_db()
        existing_user = db.execute(
            'SELECT id FROM users WHERE username = ? OR email = ?',
            (username, email)
        ).fetchone()

        if existing_user:
            flash('Username or email already exists.', 'error')
            db.close()
            return render_template('signup.html')

        # Create new user
        password_hash = generate_password_hash(password)
        try:
            db.execute(
                'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                (username, email, password_hash)
            )
            db.commit()
            flash('Account created successfully! Please sign in.', 'success')
            db.close()
            return redirect(url_for('signin'))
        except sqlite3.Error as e:
            flash(f'An error occurred: {str(e)}', 'error')
            db.close()
            return render_template('signup.html')

    return render_template('signup.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    """User signin page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            flash('Please enter both username and password.', 'error')
            return render_template('signin.html')

        db = get_db()
        user = db.execute(
            'SELECT * FROM users WHERE username = ?',
            (username,)
        ).fetchone()
        db.close()

        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash(f'Welcome back, {user["username"]}!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'error')
            return render_template('signin.html')

    return render_template('signin.html')

@app.route('/signout')
def signout():
    """Sign out the current user"""
    session.clear()
    flash('You have been signed out.', 'info')
    return redirect(url_for('signin'))

@app.route('/home')
@login_required
def home():
    """Home page with image upload"""
    # Get user's recent captions
    db = get_db()
    recent_captions = db.execute(
        'SELECT * FROM captions WHERE user_id = ? ORDER BY created_at DESC LIMIT 10',
        (session['user_id'],)
    ).fetchall()
    db.close()

    return render_template('home.html', recent_captions=recent_captions)

@app.route('/generate_caption', methods=['POST'])
@login_required
def generate_caption():
    """Generate caption for uploaded image"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg'}), 400

    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{session['user_id']}_{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Generate caption using the model
        model = get_caption_model()
        caption, confidence = model.generate_caption(
            filepath,
            method='beam_search',
            beam_width=5,
            return_confidence=True
        )

        # Save to database
        db = get_db()
        db.execute(
            'INSERT INTO captions (user_id, image_path, caption, confidence) VALUES (?, ?, ?, ?)',
            (session['user_id'], unique_filename, caption, confidence)
        )
        db.commit()
        db.close()

        return jsonify({
            'success': True,
            'caption': caption,
            'confidence': round(confidence * 100, 2),
            'image_url': url_for('static', filename=f'uploads/{unique_filename}')
        })

    except Exception as e:
        return jsonify({'error': f'Error generating caption: {str(e)}'}), 500

@app.route('/history')
@login_required
def history():
    """View caption history"""
    db = get_db()
    captions = db.execute(
        'SELECT * FROM captions WHERE user_id = ? ORDER BY created_at DESC',
        (session['user_id'],)
    ).fetchall()
    db.close()

    return render_template('history.html', captions=captions)

if __name__ == '__main__':
    # Initialize database on first run
    if not os.path.exists(app.config['DATABASE']):
        init_db()
        print('Database initialized!')

    print('Starting Flask application...')
    print('Note: The image captioning model will be loaded on first caption generation.')
    app.run(debug=True, host='0.0.0.0', port=5000)
