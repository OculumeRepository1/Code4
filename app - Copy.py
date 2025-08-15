from flask import (
    Flask, render_template, redirect, url_for,
    request, flash, session, send_from_directory, jsonify, Response, stream_with_context
)
from flask_login import (
    LoginManager, UserMixin, login_user,
    login_required, logout_user, current_user
)
import sqlite3
import bcrypt
import os
import pandas as pd
from util import load_data, count_data, pie_plot_2, bar_plot
import json
import plotly
import plotly.express as px
from util import ragmodel 
import threading
import time
import re
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Lock, Thread
import csv
from queue import Queue
from datetime import datetime
import threading
# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global variables
project_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_dir, "data.csv")
save_dir = datetime.now().strftime("%Y-%m-%d")
save_dir_dataset = os.path.join("Dataset", save_dir[0:4] + save_dir[5:7] + save_dir[8:])
# Initialize RAG model once at startup
print(data_path)

# Thread-safe data structures
data_lock = Lock()
clients = []
latest_detection_row = {}
last_row_count = 0
last_position = 0
last_modified = None

# CSV Monitoring
class CSVChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        global last_row_count, latest_detection_row
        if event.src_path.endswith(data_path):
            try:
                df = pd.read_csv(event.src_path)
                if len(df) > 0:  # Check if there are any rows
                    first_row = df.iloc[0]  # Get the first row
                    image_path = first_row.get('Image', '')  # Get the Image path
                    print(f'Image path: {image_path}')
                    if image_path:  # Only update if we have an image path
                        with data_lock:
                            latest_detection_row = {
                                'Image': image_path.replace('/', '\\'),  # Normalize path separators for Windows
                                'Detection': first_row.get('Detection', ''),
                                'Location': first_row.get('Location', ''),
                                'Date': first_row.get('Date', ''),
                                'Time': first_row.get('Time', '')
                            }
                            print(f"New detection image: {latest_detection_row['Image']}")
            except pd.errors.EmptyDataError:
                print("Warning: Empty CSV file detected")
            except pd.errors.ParserError:
                print("Warning: Error parsing CSV file")
            except Exception as e:
                print(f"Error processing CSV file: {str(e)}")

def start_csv_observer():
    event_handler = CSVChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(data_path), recursive=False)
    observer.start()
    print("CSV Watchdog started.")
    return observer

def monitor_csv():
    global last_position, last_modified
    print("CSV monitoring thread started")
    
    while True:
        try:
            current_modified = os.path.getmtime(data_path)  # Use absolute path
            
            if current_modified != last_modified:
                with open(data_path, 'r') as f:  # Use absolute path
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    current_length = len(rows)
                    if current_length > last_position:
                        num=current_length-last_position
                        new_rows = rows[0:num]
                        with data_lock:
                            for row in new_rows:
                                for client in clients:
                                    client.put(row)
                        
                        last_position = current_length
                        last_modified = current_modified
                
                time.sleep(0.5)  # Short sleep after processing changes
            else:
                time.sleep(1)  # Longer sleep if no changes
                
        except Exception as e:
            print(f"Monitoring error: {str(e)}")
            time.sleep(5)  # Wait before retrying after error

def event_stream():
    while True:
        with data_lock:
            if latest_detection_row:
                yield f"data: {json.dumps(latest_detection_row)}\n\n"
        time.sleep(1)

# Flask-Login Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database Functions
def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)
        return None

def create_users_table():
    conn = create_connection(os.path.join(project_dir, "users.db"))
    if conn:
        try:
            cur = conn.cursor()
            cur.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL UNIQUE,
                    password TEXT NOT NULL
                )
            ''')
            conn.commit()
            print("Users table checked/created successfully!")
        except sqlite3.Error as e:
            print(e)
        finally:
            conn.close()

# User Model
class User(UserMixin):
    def __init__(self, user_id, username, email, password):
        self.id = user_id
        self.username = username
        self.email = email
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    conn = create_connection(os.path.join(project_dir, "users.db"))
    if conn:
        cur = conn.cursor()
        cur.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user_data = cur.fetchone()
        conn.close()
        if user_data:
            return User(*user_data)
    return None

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']

        hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

        conn = create_connection(os.path.join(project_dir, "users.db"))
        if conn:
            try:
                cur = conn.cursor()
                cur.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                            (username, email, hashed_password))
                conn.commit()
                flash('Sign up successful. Please login.', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('Username or email already exists.', 'danger')
            finally:
                conn.close()
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        identifier = request.form.get('identifier')
        password = request.form.get('password')
        
        if not identifier or not password:
            flash('Please fill in both identifier and password.', 'danger')
            return render_template('login.html')

        conn = create_connection(os.path.join(project_dir, "users.db"))
        if conn:
            try:
                cur = conn.cursor()
                cur.execute('SELECT * FROM users WHERE username = ? OR email = ?', (identifier, identifier))
                user_data = cur.fetchone()
                
                if user_data and bcrypt.checkpw(password.encode(), user_data[3]):
                    user = User(*user_data)
                    login_user(user)
                    session['login'] = 'Yes'
                    flash('Login successful!', 'success')
                    return redirect(url_for('dashboard'))
                else:
                    flash('Invalid username, email, or password.', 'danger')
            except Exception as e:
                print(f"Login error: {str(e)}")
                flash('An error occurred during login.', 'danger')
            finally:
                conn.close()
    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    try:
        data_log = load_data(data_path)
        data_log['Date'] = pd.to_datetime(data_log['Date']).dt.date

        min_date = data_log['Date'].min()
        max_date = data_log['Date'].max()
        date_range = (min_date, max_date)

        locations = data_log['Location'].unique().tolist()
        classes = data_log['Detection'].unique().tolist()

        start_date = min_date
        end_date = max_date
        selected_locations = locations
        selected_classes = classes

        if request.method == 'POST':
            start_date_input = request.form.get('start_date', str(min_date))
            end_date_input = request.form.get('end_date', str(max_date))

            start_date = pd.to_datetime(start_date_input).date()
            end_date = pd.to_datetime(end_date_input).date()

            selected_locations = request.form.getlist('locations') or locations
            selected_classes = request.form.getlist('classes') or classes

        condition = (
            (data_log['Date'] >= start_date) &
            (data_log['Date'] <= end_date) &
            (data_log['Location'].isin(selected_locations)) &
            (data_log['Detection'].isin(selected_classes))
        )
        filtered_data = data_log[condition]

        if filtered_data.empty:
            flash('No data available for the selected filters.', 'warning')
            detection_counts = pd.DataFrame(columns=['Detection', 'count'])
        else:
            detection_counts = count_data(filtered_data, 'Detection')

        total_detections = filtered_data.shape[0]
        pie_chart = pie_plot_2(filtered_data, detection_counts, 'Detection')
        bar_chart = bar_plot(filtered_data, detection_counts, 'Detection', 'Detection Classes', 'Detection Data')
        data_table_html = filtered_data.to_html(classes='table table-striped table-bordered table-sm', index=False)

        path_to_html = os.path.join("static", "Incident_2024.html")
        try:
            with open(path_to_html, 'r') as f:
                html_data = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            html_data = f"<p>Error reading file: {e}</p>"

        return render_template(
            'dashboard.html',
            date_range=date_range,
            locations=locations,
            classes=classes,
            pie_chart=pie_chart.to_html(),
            bar_chart=bar_chart.to_html(),
            html_data=html_data,
            data_table=data_table_html,
            total_detections=total_detections,
            selected_locations=selected_locations,
            selected_classes=selected_classes,
            start_date=start_date,
            end_date=end_date
        )
    except Exception as e:
        print(f"Dashboard error: {str(e)}")
        flash('An error occurred while loading the dashboard.', 'danger')
        return redirect(url_for('home'))

@app.route('/LiveStreamAnalytics')
@login_required
def live_stream_analytics():
    try:
        # Load data with error handling
        try:
            df = pd.read_csv(data_path)
            #df=df[df['Detection'] != 'NonViolence']
            if df.empty:
                flash('No data found in the CSV file.', 'warning')
                return render_template('LiveStreamAnalytics.html', 
                                    images=[], 
                                    total_detections=0, 
                                    active_detections=0, 
                                    severity_counts={'critical': 0, 'medium': 0, 'low': 0})
            
            # Ensure required columns exist
            required_columns = ['Image', 'Detection', 'Location', 'Date', 'Time', 'Status']
            for col in required_columns:
                if col not in df.columns:
                    flash(f'Missing required column: {col}', 'danger')
                    return render_template('LiveStreamAnalytics.html', 
                                         images=[], 
                                         total_detections=0, 
                                         active_detections=0, 
                                         severity_counts={'critical': 0, 'medium': 0, 'low': 0})
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            flash('Error loading data from CSV file.', 'danger')
            return render_template('LiveStreamAnalytics.html', 
                                 images=[], 
                                 total_detections=0, 
                                 active_detections=0, 
                                 severity_counts={'critical': 0, 'medium': 0, 'low': 0})

        # Get current date in YYYYMMDD format
        current_date = datetime.now().strftime('%Y%m%d')
        
        # Filter for images with status 'pending' or empty status
        pending_df = df[df['Status'].isin(['pending', ''])].copy()
        
        # Debug output
        print("\nPending Images:")
        print(f"Total pending images: {len(pending_df)}")
        print("Pending image paths:")
        for i, row in pending_df.head().iterrows():
            print(f"Image: {row['Image']}, Status: {row['Status']}")
        
        # Debug output
        print("\nPending Images:")
        print(f"Total pending images: {len(pending_df)}")
        print("Pending image paths:")
        for i, row in pending_df.head().iterrows():
            print(f"Image: {row['Image']}, Status: {row['Status']}")
        
        # Normalize and process image paths
        def process_image_path(path):
            # Convert Windows path to web-friendly format
            if path and len(path) > 1 and path[1] == ':':
                # Keep the path but replace backslashes with forward slashes
                return path.replace('\\', '/')
            # For relative paths, keep as is
            return path
        
        # Process image paths
        pending_df['Image'] = pending_df['Image'].apply(process_image_path)
        
        # Debug final paths
        print("\nFinal Image Paths:")
        print(f"Total images found: {len(pending_df)}")
        print("Sample image paths:")
        for i, row in pending_df.head().iterrows():
            print(f"Image: {row['Image']}, Status: {row['Status']}")
        
        # Calculate severity counts
        critical_detections = ['gun', 'knife', 'pepper spray']
        medium_detections = ['fighting', 'Violence', 'chair weapon']
        
        pending_df['Severity'] = pending_df['Detection'].apply(
            lambda x: 'critical' if str(x).lower() in critical_detections 
                      else 'medium' if str(x).lower() in medium_detections 
                      else 'low'
        )
        
        severity_counts = {
            'critical': len(pending_df[pending_df['Severity'] == 'critical']),
            'medium': len(pending_df[pending_df['Severity'] == 'medium']),
            'low': len(pending_df[pending_df['Severity'] == 'low'])
        }
        
        
        return render_template('LiveStreamAnalytics.html',
            images=pending_df.to_dict('records'),
            total_detections=len(pending_df),
            active_detections=len(pending_df[pending_df['Status'] == 'pending']),
            severity_counts=severity_counts
        )
    except Exception as e:
        print(f"Error in images route: {str(e)}")
        return render_template('LiveStreamAnalytics.html',
                             images=[], 
                             total_detections=0, 
                             active_detections=0, 
                             severity_counts={'critical': 0, 'medium': 0, 'low': 0})

@app.route('/<path:filename>')
def send_image(filename):
    try:
        #f = filename[4:]
        print(f"Requested image: {filename}")
        # Normalize path separators for Windows
        filename = filename.replace('/', os.sep)
        print(f"File image path: {filename}")
        # Construct full path
        full_path = os.path.join(project_dir, filename)
        
        
        if os.path.exists(full_path):
            print(f"Serving image from: {full_path}")
            return send_from_directory(os.path.dirname(full_path), os.path.basename(full_path))
        else:
            print(f"Image not found at: {filename}")
            # Return a placeholder image if not found
            return send_from_directory(os.path.join(project_dir, 'static', 'images'), 'placeholder.jpg')
    except Exception as e:
        print(f"Error serving image: {str(e)}")
        # Return a placeholder image on error
        return send_from_directory(os.path.join(project_dir, 'static', 'images'), 'placeholder.jpg')

@app.route('/handle-image', methods=['POST'])
@login_required
def handle_image():
    try:
        data = request.get_json()
        image_path = data.get('image_path')  # Just the filename
        action = data.get('action')
        print(f"Handling image: {image_path} with action: {action}")
        
        if not image_path or not action:
            return jsonify({'success': False, 'error': 'Missing parameters'}), 400

        # Validate action
        if action not in ['accept', 'reject']:
            return jsonify({'success': False, 'error': 'Invalid action'}), 400

        # Normalize path for Windows
        image_path = os.path.normpath(image_path)
        print(f"Normalized path: {image_path}")
        
        # Read the CSV file using the correct path
        df = pd.read_csv(data_path)
        #df=df[df['Detection'] != 'NonViolence']
        # Normalize paths for comparison
        normalized_path = os.path.normpath(image_path)
        df['Image'] = df['Image'].apply(lambda x: os.path.normpath(x))
        
        # Find the row with the exact matching image path
        mask = df['Image'] == normalized_path
        print(f"Looking for: {normalized_path}")
        print(f"CSV contents: {df['Image'].tolist()}")
        print(f"Mask: {mask}")
        if not mask.any():
            return jsonify({'success': False, 'error': 'Image not found in database'}), 404

        # Update the Status column
        df.loc[mask, 'Status'] = action
        print(df)
        # Save the updated CSV
        df.to_csv(data_path, index=False)

        return jsonify({
            'success': True,
            'message': f'Image {action}ed successfully',
            'image_path': image_path
        })
        
    except Exception as e:
        print(f"Error in handle_image: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/handle-images', methods=['POST'])
@login_required
def handle_images():
    try:
        data = request.get_json()
        image_paths = data.get('image_paths', [])
        action = data.get('action')
        #print(f"Handling {len(image_paths)} images with action: {action}")
        
        if not image_paths or not action:
            return jsonify({'success': False, 'error': 'Missing parameters'}), 400
        
        if action not in ['accept', 'reject']:
            return jsonify({'success': False, 'error': 'Invalid action'}), 400

        # Normalize paths for comparison
        normalized_paths = [os.path.normpath(path) for path in image_paths]
        
        # Read the CSV file
        df = pd.read_csv(data_path)
        df['Image'] = df['Image'].apply(lambda x: os.path.normpath(x))
        
        # Update all matching rows
        mask = df['Image'].isin(normalized_paths)
        #print(f"Updating {mask.sum()} rows")
        
        if mask.any():
            df.loc[mask, 'Status'] = action
            df.to_csv(data_path, index=False)
            return jsonify({
                'success': True,
                'message': f'Successfully updated {mask.sum()} images',
                'updated_images': df.loc[mask, 'Image'].tolist()
            })
        else:
            return jsonify({'success': False, 'error': 'No matching images found'}), 404
            
    except Exception as e:
        print(f"Error in handle_images: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# @login_required
# def handle_image():
#     try:
#         data = request.get_json()
#         image_name = data.get('image_name')  # Just the filename
#         action = data.get('action')
#         print(image_name)
#         if not image_name or not action:
#             return jsonify({'success': False, 'error': 'Missing parameters'}), 400

#         # Validate action
#         if action not in ['accept', 'reject']:
#             return jsonify({'success': False, 'error': 'Invalid action'}), 400

#         # Get current date folder
#         #current_date = datetime.now().strftime("%Y%m%d")
#         image_dir = os.path.join(project_dir, save_dir_dataset)
#         full_image_path = os.path.join(image_dir, image_name)

#         # Load CSV with proper NA handling
#         df = pd.read_csv(data_path, keep_default_na=False)
        
#         # Get the first row of the CSV
#         first_row = df.iloc[0]
        
#         # Ensure Image column exists
#         if 'Image' not in df.columns:
#             return jsonify({'success': False, 'error': 'Image column missing in CSV'}), 500

#         # Get the image path from the first row
#         image_path = first_row['Image']
#         print(image_path)
#         # Create safe comparison
#         df['ImageName'] = df['Image'].apply(lambda x: os.path.basename(str(x)))
#         mask = df['ImageName'] == image_name
        
#         if not mask.any():
#             return jsonify({'success': False, 'error': 'Image not found in database'}), 404

#         # For reject action, delete the image file
#         if action == 'reject':
#             if os.path.exists(image_path):
#                 try:
#                     os.remove(image_path)
#                 except Exception as e:
#                     return jsonify({
#                         'success': False,
#                         'error': f'Failed to delete image: {str(e)}'
#                     }), 500

#         # For reject action, delete the image file
#         if action == 'reject':
#             if os.path.exists(full_image_path):
#                 try:
#                     os.remove(full_image_path)
#                 except Exception as e:
#                     return jsonify({
#                         'success': False,
#                         'error': f'Failed to delete image: {str(e)}'
#                     }), 500

#         # Remove the row from DataFrame
#         df = df[~mask]
        
#         # Save updated CSV
#         df.to_csv(data_path, index=False)

#         return jsonify({
#             'success': True,
#             'message': f'Image {action}ed successfully',
#             'image_name': image_name
#         })
        
#     except Exception as e:
#         return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/monitor-dataset')
def monitor_dataset():
    def generate():
        while True:
            try:
                image_dir = os.path.join(project_dir, 'Dataset')
                images = [f for f in os.listdir(image_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                yield f"data: {json.dumps({'images': images})}\n\n"
                time.sleep(5)
            except Exception as e:
                print(f"Error monitoring dataset: {str(e)}")
                time.sleep(5)
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('login', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    if 'chat_history' not in session:
        session['chat_history'] = []

    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'Message is required'}), 400

    session['chat_history'].append({"role": "user", "content": user_message})

    if qa:
        try:
            ai_reply = qa(user_message)["result"]
        except Exception as e:
            print(f"Error processing chat: {str(e)}")
            ai_reply = "I'm having trouble processing your request. Please try again later."
            session['chat_history'].append({"role": "assistant", "content": ai_reply})
        session.modified = True
        return jsonify({'reply': ai_reply})
    else:
        ai_reply = "Error: RAG model not available"

    session['chat_history'].append({"role": "assistant", "content": ai_reply})
    session.modified = True
    return jsonify({'reply': ai_reply})

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    return jsonify(session.get('chat_history', []))

@app.route('/data-stream')
def data_stream():
    def generate():
        client_queue = Queue()
        with data_lock:
            clients.append(client_queue)
        print(f"New client connected (Total: {len(clients)})")
        
        try:
            while True:
                if not client_queue.empty():
                    data = client_queue.get()
                    yield f"data: {json.dumps(data)}\n\n"
                time.sleep(0.1)
        finally:
            with data_lock:
                clients.remove(client_queue)
            print(f"Client disconnected (Remaining: {len(clients)})")
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/refresh-data', methods=['POST'])
@login_required
def refresh_data():
    try:
        data_log = load_data(data_path)
        data_log['Date'] = pd.to_datetime(data_log['Date']).dt.date
        
        filters = request.get_json()
        condition = (
            (data_log['Date'] >= pd.to_datetime(filters['start_date']).date()) &
            (data_log['Date'] <= pd.to_datetime(filters['end_date']).date()) &
            (data_log['Location'].isin(filters['locations'])) &
            (data_log['Detection'].isin(filters['classes']))
        )
        filtered_data = data_log[condition]

        detection_counts = count_data(filtered_data, 'Detection') if not filtered_data.empty else pd.DataFrame()
        
        return jsonify({
            'pie': pie_plot_2(filtered_data, detection_counts, 'Detection').to_html(),
            'bar': bar_plot(filtered_data, detection_counts, 'Detection', 
                          'Detection Classes', 'Detection Data').to_html(),
            'table': filtered_data[['Date', 'Time', 'Detection', 'Location', 'Summary']].to_html(
                classes='table table-striped table-bordered table-sm', 
                index=False
            ),
            'total': filtered_data.shape[0],
        })
    except Exception as e:
        return jsonify(error=str(e)), 500

# Run the App
if __name__ == '__main__':
    data = load_data(data_path)
    #qa = ragmodel(data)
    create_users_table()
    # Start CSV file watcher in a separate thread
    # Start CSV monitoring
    # Start CSV monitoring thread
    csv_thread = threading.Thread(target=monitor_csv, daemon=True)
    csv_thread.start()

    app.run()
    observer.join()