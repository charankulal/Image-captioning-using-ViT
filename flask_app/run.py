"""Simple script to run the Flask application"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and run the app
from flask_app.app import app, init_db

if __name__ == '__main__':
    # Initialize database if it doesn't exist
    if not os.path.exists('flask_app/database.db'):
        print('Initializing database...')
        init_db()
        print('Database initialized!')

    print('\n' + '='*80)
    print('Image Captioning Web Application')
    print('='*80)
    print(f'Server starting on http://localhost:5000')
    print('Note: The image captioning model will load on first caption generation.')
    print('This may take a few seconds on first use.')
    print('='*80 + '\n')

    app.run(debug=True, host='0.0.0.0', port=5000)
