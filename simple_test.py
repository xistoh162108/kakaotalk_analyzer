"""Simple test to verify basic functionality without external dependencies"""

import sys
import os
from datetime import datetime, timedelta
import csv
import json
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def generate_simple_test_csv():
    """Generate minimal test CSV"""
    
    test_data = [
        ['user', 'message', 'datetime'],
        ['김민수', '안녕하세요!', '2024-01-01 10:00:00'],
        ['이지은', '네, 안녕하세요! ㅎㅎ', '2024-01-01 10:01:00'],
        ['김민수', '오늘 날씨 좋네요', '2024-01-01 10:02:00'],
        ['이지은', '정말 그러네요!', '2024-01-01 10:03:00'],
        ['박서준', '뭐 하고 계세요?', '2024-01-01 11:00:00'],
        ['김민수', '그냥 일하고 있어요', '2024-01-01 11:01:00'],
        ['이지은', '저도 일하는 중이에요', '2024-01-01 11:02:00'],
        ['박서준', '점심 뭐 드실래요?', '2024-01-01 12:00:00'],
        ['김민수', '김치찌개 어때요?', '2024-01-01 12:01:00'],
        ['이지은', '좋아요! ㅋㅋㅋ', '2024-01-01 12:02:00']
    ]
    
    with open('test.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(test_data)
    
    print("✅ test.csv 생성 완료!")
    return 'test.csv'

def test_imports():
    """Test if all modules can be imported"""
    
    test_results = []
    
    try:
        from kakao_analyzer.config import Config
        test_results.append("✅ Config import successful")
    except Exception as e:
        test_results.append(f"❌ Config import failed: {e}")
    
    try:
        from kakao_analyzer.utils import setup_logger
        test_results.append("✅ Utils import successful") 
    except Exception as e:
        test_results.append(f"❌ Utils import failed: {e}")
    
    try:
        from kakao_analyzer.__main__ import main
        test_results.append("✅ Main module import successful")
    except Exception as e:
        test_results.append(f"❌ Main module import failed: {e}")
    
    return test_results

def test_config():
    """Test configuration creation"""
    
    try:
        from kakao_analyzer.config import Config
        config = Config()
        
        # Check if essential attributes exist
        essential_attrs = [
            'ollama_model', 'embed_model', 'window_minutes', 
            'topic_window_size', 'similarity_threshold'
        ]
        
        for attr in essential_attrs:
            if not hasattr(config, attr):
                return f"❌ Config missing attribute: {attr}"
        
        return "✅ Config creation and validation successful"
    
    except Exception as e:
        return f"❌ Config test failed: {e}"

def test_output_structure():
    """Test if CLI can create output structure"""
    
    try:
        from kakao_analyzer.utils import create_output_directories
        
        test_output = Path("test_output")
        paths = create_output_directories(test_output)
        
        # Check if all required directories were created
        required_dirs = ['summary', 'basics', 'context', 'fun', 'figures', 'reports', 'logs']
        
        for dir_name in required_dirs:
            if dir_name not in paths:
                return f"❌ Missing output directory: {dir_name}"
            
            if not paths[dir_name].exists():
                return f"❌ Directory not created: {dir_name}"
        
        return "✅ Output directory structure creation successful"
    
    except Exception as e:
        return f"❌ Output structure test failed: {e}"

def main():
    """Run simple tests"""
    
    print("="*50)
    print("카카오톡 분석기 기본 기능 테스트")
    print("="*50)
    
    # Generate test data
    print("\n1. 테스트 데이터 생성...")
    csv_path = generate_simple_test_csv()
    
    # Test imports
    print("\n2. 모듈 import 테스트...")
    import_results = test_imports()
    for result in import_results:
        print(f"   {result}")
    
    # Test config
    print("\n3. 설정 테스트...")
    config_result = test_config()
    print(f"   {config_result}")
    
    # Test output structure
    print("\n4. 출력 구조 테스트...")
    output_result = test_output_structure()
    print(f"   {output_result}")
    
    # Test CLI help
    print("\n5. CLI 도움말 테스트...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, '-m', 'kakao_analyzer', '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("   ✅ CLI help command successful")
        else:
            print(f"   ❌ CLI help failed: {result.stderr}")
    except Exception as e:
        print(f"   ❌ CLI help test failed: {e}")
    
    # Summary
    print("\n" + "="*50)
    print("테스트 완료!")
    print("="*50)
    
    if Path(csv_path).exists():
        print(f"✅ 테스트 데이터: {csv_path}")
    
    if Path("test_output").exists():
        print("✅ 출력 디렉토리: test_output/")
    
    print("\n다음 단계:")
    print("1. pip install -r requirements.txt (의존성 설치)")
    print("2. python -m kakao_analyzer test.csv test_output (분석 실행)")
    print("3. test_output/reports/analysis_report.md (결과 확인)")

if __name__ == "__main__":
    main()