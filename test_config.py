#!/usr/bin/env python3
"""
BirdVision India Configuration Test Script
Tests the configuration and basic functionality without starting the full app.
"""

import os
import sys
import importlib.util

def test_configuration():
    """Test configuration loading"""
    try:
        from config import Config
        print("✅ Configuration loaded successfully")
        
        # Test configuration values
        assert hasattr(Config, 'BIRD_CLASSES'), "BIRD_CLASSES not found in config"
        assert len(Config.BIRD_CLASSES) == 8, f"Expected 8 bird classes, found {len(Config.BIRD_CLASSES)}"
        assert hasattr(Config, 'ROBOFLOW_API_KEY'), "ROBOFLOW_API_KEY not found in config"
        
        print(f"✅ Found {len(Config.BIRD_CLASSES)} bird species in configuration")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_dependencies():
    """Test if required dependencies are available"""
    required_packages = [
        'flask', 'cv2', 'supervision', 'numpy', 'tensorflow', 
        'librosa', 'matplotlib', 'PIL', 'plotly', 'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT FOUND")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All required dependencies are available")
    return True

def test_file_structure():
    """Test if required files and directories exist"""
    required_files = [
        'app.py',
        'config.py', 
        'requirements.txt',
        'static/style.css',
        'templates/base.html'
    ]
    
    required_dirs = [
        'static',
        'templates',
        'static/uploads'
    ]
    
    missing_items = []
    
    # Check files
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            missing_items.append(file_path)
    
    # Check directories
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - NOT FOUND")
            missing_items.append(dir_path)
    
    if missing_items:
        print(f"\n❌ Missing items: {', '.join(missing_items)}")
        return False
    
    print("✅ All required files and directories found")
    return True

def test_models():
    """Test if model files exist"""
    from config import Config
    
    model_files = [
        Config.AUDIO_MODEL_PATH,
        Config.METADATA_CSV_PATH
    ]
    
    missing_models = []
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"✅ {model_file} ({size_mb:.1f} MB)")
        else:
            print(f"⚠️  {model_file} - NOT FOUND")
            missing_models.append(model_file)
    
    if missing_models:
        print(f"\n⚠️  Missing model files: {', '.join(missing_models)}")
        print("The app may not work properly without these files.")
        return False
    
    print("✅ All model files found")
    return True

def main():
    """Run all tests"""
    print("🧪 BirdVision India Configuration Test")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Dependencies", test_dependencies), 
        ("File Structure", test_file_structure),
        ("Model Files", test_models)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        emoji = "✅" if result else "❌"
        print(f"{emoji} {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! The application should work correctly.")
        print("You can now run: python app.py")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues before running the app.")
        return 1

if __name__ == "__main__":
    exit(main())