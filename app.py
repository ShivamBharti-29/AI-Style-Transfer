import os
import torch
import datetime  # Added for gallery timestamps
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired
from PIL import Image
from torchvision import transforms
import io

# Import your existing AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = VGGEncoder('vgg_normalised.pth').to(device)
decoder = Decoder().to(device)
# Use your specific path as provided in your original code
decoder.load_state_dict(torch.load('adain-style-transfer/experiment/final_model/decoder_2.pth'))

encoder.eval()
decoder.eval()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)
    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)
        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)
        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats
        stylized_image = decoder(stylized_feats)
    return stylized_image

def save_image(image, path):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)

# --- ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = secure_filename(form.content.data.filename)
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                form.content_path.data = content_filename
        else:
            content_filename = form.content_path.data

        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = secure_filename(form.style.data.filename)
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                form.style_path.data = style_filename
        else:
            style_filename = form.style_path.data

        if content_filename and style_filename:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
            try:
                content_image = Image.open(content_path).convert('RGB')
                style_image = Image.open(style_path).convert('RGB')
                alpha = float(form.alpha.data)
                stylized_image = style_transfer(content_image, style_image, encoder, decoder, alpha, device)
                result_filename = 'stylized_' + content_filename
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                save_image(stylized_image, result_path)
                result_image = result_filename
            except Exception as e:
                error = str(e)
    return render_template('index.html', form=form, result_image=result_image, content_image=content_filename, style_image=style_filename, error=error)



@app.route('/gallery')
def gallery():
    images = []
    # Ensure this path matches where your stylized images are saved
    output_folder = os.path.join('static', 'outputs') 
    
    if os.path.exists(output_folder):
        # 1. Get all files and filter for your generated images
        all_files = os.listdir(output_folder)
        for filename in all_files:
            if filename.startswith('stylized_'):
                filepath = os.path.join(output_folder, filename)
                
                # 2. Get generation time
                timestamp = os.path.getctime(filepath)
                dt = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
                
                images.append({
                    'url': filename,
                    'time': dt
                })
    
    # 3. Sort newest first
    images.sort(key=lambda x: x['time'], reverse=True)
    
    return render_template('gallery.html', images=images)

# NEW: Route to handle image deletion
@app.route('/delete/<filename>', methods=['POST'])
def delete_art(filename):
    try:
        file_path = os.path.join('static', 'outputs', filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return {"success": True}
        return {"success": False}, 404
    except Exception:
        return {"success": False}, 500

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)