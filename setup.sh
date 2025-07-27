#!/bin/bash

# BirdVision India Setup Script
echo "🐦 Setting up BirdVision India..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create uploads directory
echo "📁 Creating upload directory..."
mkdir -p static/uploads

# Check if models exist
echo "🔍 Checking for required models..."
if [ ! -f "model_16_val_loss_0.1970.keras" ]; then
    echo "⚠️  Audio model not found: model_16_val_loss_0.1970.keras"
    echo "   Please ensure this file is present in the project directory"
fi

if [ ! -f "train_metadata - Copy.csv" ]; then
    echo "⚠️  Metadata file not found: train_metadata - Copy.csv"
    echo "   Please ensure this file is present for map visualization"
fi

# Environment variables setup
echo "🔐 Setting up environment variables..."
if [ ! -f ".env" ]; then
    cat > .env << EOL
# BirdVision India Environment Variables
ROBOFLOW_API_KEY=your-roboflow-api-key-here
SECRET_KEY=birdvision-india-secret-key-2024
EOL
    echo "📝 Created .env file. Please update with your actual API keys."
else
    echo "✅ .env file already exists"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Update the .env file with your Roboflow API key"
echo "2. Ensure model files are present"
echo "3. Run: source venv/bin/activate"
echo "4. Run: python app.py"
echo "5. Open: http://localhost:5000"
echo ""
echo "For help, check the README.md file"