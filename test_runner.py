"""Test runner for Kakao analyzer"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path
import logging
import json

from kakao_analyzer.config import Config
from kakao_analyzer.pipeline import KakaoAnalysisPipeline
from kakao_analyzer.utils import setup_logger
from kakao_analyzer.report import ReportGenerator


def generate_test_data(num_messages: int = 1000, output_path: str = "test.csv") -> str:
    """Generate realistic test data for Kakao analysis"""
    
    # Sample users with Korean names
    users = ["김민수", "이지은", "박서준", "최유진", "정다은", "장혁진"]
    
    # Sample Korean conversation starters and responses
    conversation_starters = [
        "안녕하세요!", "오늘 날씨 좋네요", "뭐 하고 계세요?", "점심 뭐 드셨어요?",
        "요즘 어떻게 지내세요?", "이번 주말에 뭐 하세요?", "새로운 카페 가봤어요?",
        "영화 추천해주세요", "맛있는 식당 아시나요?", "취미가 뭐예요?"
    ]
    
    responses = [
        "네, 안녕하세요! ㅎㅎ", "정말 그러네요!", "그냥 일하고 있어요", "김치찌개 먹었어요",
        "바쁘게 지내고 있어요", "아직 계획 없어요", "아직 안 가봤어요 ㅠㅠ",
        "기생충 어때요?", "홍대 쪽에 좋은 데 있어요", "독서하는 거 좋아해요",
        "ㅋㅋㅋㅋ 웃겨요", "정말요?", "대박!", "완전 좋아요", "그럴 수도 있겠네요",
        "맞아요!", "저도 그렇게 생각해요", "아 그래요?", "신기하네요", "좋은 생각이에요"
    ]
    
    # Generate conversations with realistic patterns
    messages = []
    current_time = datetime.now() - timedelta(days=30)  # Start 30 days ago
    
    for i in range(num_messages):
        # Realistic time progression
        time_increment = random.randint(1, 300)  # 1-300 minutes between messages
        current_time += timedelta(minutes=time_increment)
        
        # Select user (with some users being more active)
        user_weights = [0.3, 0.25, 0.2, 0.15, 0.07, 0.03]  # Realistic distribution
        user = np.random.choice(users, p=user_weights)
        
        # Generate message content
        if i == 0 or random.random() < 0.1:  # Conversation starter
            message = random.choice(conversation_starters)
        else:
            message = random.choice(responses)
        
        # Add some variety with emojis and reactions
        if random.random() < 0.3:
            reactions = ["ㅋㅋㅋ", "ㅎㅎ", "ㅠㅠ", "ㅜㅜ", "ㄱㄱ", "ㅇㅋ"]
            message += " " + random.choice(reactions)
        
        # Some longer messages occasionally
        if random.random() < 0.05:
            extensions = [
                " 정말 재미있었어요! 다음에 또 해요",
                " 시간이 빨리 지나가는 것 같아요",
                " 요즘 이런 일들이 많이 일어나나 봐요",
                " 좋은 경험이었던 것 같네요"
            ]
            message += random.choice(extensions)
        
        messages.append({
            'user': user,
            'message': message,
            'datetime': current_time.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(messages)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"Generated {num_messages} test messages saved to {output_path}")
    return output_path


def run_full_test_suite():
    """Run comprehensive test suite"""
    
    # Setup
    logger = setup_logger("test_runner")
    config = Config()
    
    # Generate test data if it doesn't exist
    test_csv_path = Path("test.csv")
    if not test_csv_path.exists():
        logger.info("Generating test data...")
        generate_test_data(500, str(test_csv_path))  # Smaller dataset for testing
    
    # Initialize pipeline
    pipeline = KakaoAnalysisPipeline(config, logger)
    
    # Test 1: Component testing
    logger.info("Running component tests...")
    test_df = pd.read_csv(test_csv_path)
    test_df['datetime'] = pd.to_datetime(test_df['datetime'])
    
    component_results = pipeline.test_pipeline_components(test_df)
    
    # Test 2: Full pipeline test
    logger.info("Running full pipeline test...")
    output_dir = Path("test_output")
    
    pipeline_results = pipeline.run_full_analysis(test_csv_path, output_dir)
    
    # Test 3: File generation verification
    logger.info("Verifying output files...")
    required_files = [
        "summary/analysis_summary.json",
        "basics/basic_statistics.json", 
        "context/topic_segments.json",
        "fun/fun_metrics.json",
        "figures/heatmap_hour_weekday.png",
        "reports/analysis_report.md",
        "logs/pipeline_metadata.json"
    ]
    
    file_checks = {}
    for file_path in required_files:
        full_path = output_dir / file_path
        file_checks[file_path] = full_path.exists()
    
    # Compile test results
    overall_results = {
        'test_file': str(test_csv_path),
        'total_tests': component_results['total_tests'] + 1,  # +1 for pipeline test
        'passed_tests': component_results['passed_tests'] + (1 if pipeline_results['success'] else 0),
        'failed_tests': component_results['failed_tests'] + (0 if pipeline_results['success'] else 1),
        'component_results': component_results,
        'pipeline_success': pipeline_results['success'],
        'file_checks': file_checks,
        'passed': component_results['passed'],
        'failed': component_results['failed']
    }
    
    if not pipeline_results['success']:
        overall_results['failed'].append({
            'name': 'full_pipeline',
            'error': pipeline_results.get('error', 'Unknown pipeline error')
        })
    else:
        overall_results['passed'].append({
            'name': 'full_pipeline',
            'description': 'Complete analysis pipeline execution'
        })
    
    overall_results['success_rate'] = (overall_results['passed_tests'] / overall_results['total_tests']) * 100
    
    # Recommendations
    recommendations = []
    if overall_results['success_rate'] == 100:
        recommendations.append("모든 테스트가 통과했습니다! 시스템이 정상적으로 작동합니다.")
    else:
        recommendations.append(f"테스트 성공률: {overall_results['success_rate']:.1f}% - 실패한 컴포넌트를 확인하세요.")
    
    if not all(file_checks.values()):
        recommendations.append("일부 출력 파일이 생성되지 않았습니다. 권한 및 디스크 공간을 확인하세요.")
    
    overall_results['recommendations'] = recommendations
    
    # Generate QA report
    if pipeline_results['success']:
        report_generator = ReportGenerator(config, logger)
        qa_report_path = output_dir / "reports" / "qa_test_report.md"
        report_generator.generate_qa_test_report(overall_results, qa_report_path)
        logger.info(f"QA test report generated: {qa_report_path}")
    
    # Save test results
    test_results_path = output_dir / "logs" / "test_results.json"
    test_results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(test_results_path, 'w', encoding='utf-8') as f:
        json.dump(overall_results, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"Test results saved to: {test_results_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("테스트 결과 요약")
    print("="*50)
    print(f"총 테스트: {overall_results['total_tests']}")
    print(f"통과: {overall_results['passed_tests']}")
    print(f"실패: {overall_results['failed_tests']}")
    print(f"성공률: {overall_results['success_rate']:.1f}%")
    print(f"파이프라인 실행: {'성공' if pipeline_results['success'] else '실패'}")
    print(f"출력 디렉토리: {output_dir}")
    
    if pipeline_results['success']:
        print(f"분석 리포트: {pipeline_results['report_path']}")
        print(f"실행 시간: {pipeline_results['duration_seconds']:.1f}초")
    
    return overall_results


if __name__ == "__main__":
    # Generate test data
    print("카카오톡 분석기 테스트 스위트")
    print("="*40)
    
    # Option to generate fresh test data
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--generate-data":
        print("새로운 테스트 데이터 생성 중...")
        generate_test_data(1000, "test.csv")
        print("테스트 데이터 생성 완료!")
    
    # Run tests
    print("테스트 실행 중...")
    results = run_full_test_suite()
    
    if results['success_rate'] == 100:
        print("\n✅ 모든 테스트가 성공적으로 완료되었습니다!")
    else:
        print(f"\n⚠️  일부 테스트가 실패했습니다. ({results['failed_tests']}/{results['total_tests']} 실패)")