# 🔥 카카오톡 대화 분석기 (Kakao Talk Analyzer)

[![PyPI version](https://badge.fury.io/py/kakao-analyzer.svg)](https://badge.fury.io/py/kakao-analyzer)
[![Python versions](https://img.shields.io/pypi/pyversions/kakao-analyzer.svg)](https://pypi.org/project/kakao-analyzer/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/kakao-analyzer)](https://pepy.tech/project/kakao-analyzer)

> 카카오톡 대화 내보내기 CSV 파일을 종합적으로 분석하는 **차세대 AI 기반** 올인원 도구

```bash
pip install kakao-analyzer
```

**🎯 카카오톡의 특성을 완벽하게 이해하는 스마트 분석기**
- 연속 메시지 자동 그룹핑 (1분 이내 메시지 통합)
- 한국어 채팅 패턴 인식 (ㅋㅋ, ㅠㅠ, 이모티콘 등)
- 시스템 메시지 자동 필터링 (삭제된 메시지, 투표, 정산 등)
- 실제 대화만 분석하여 더 정확한 인사이트 제공

## 🚀 주요 기능

### 🎯 **카카오톡 특화 기능** ⭐ NEW!
- **스마트 메시지 그룹핑**: 1분 이내 연속 메시지를 하나로 통합하여 실제 의도 파악
- **한국어 채팅 패턴 분석**: ㅋㅋ, ㅎㅎ, ㅠㅠ, ㅇㅋ 등 한국인의 채팅 습관 인식
- **시스템 메시지 필터링**: 삭제된 메시지, 투표, 정산, 입장/퇴장 알림 자동 제거
- **멘션 분석**: @ 기능을 이용한 사용자 언급 패턴과 네트워크 분석
- **감정 표현 분석**: 웃음, 슬픔, 동의 표현 등의 감정 패턴 추적

### 📊 기본 통계 분석
- **사용자별 활동 통계**: 메시지 수, 참여율, 일평균 활동량
- **시간적 패턴 분석**: 시간대별, 요일별, 월별 활동 분포  
- **단어 빈도 분석**: 한국어 형태소 분석 및 키워드 추출
- **메시지별 단어 수 통계**: 원본 및 필터링된 단어 수 분석 (평균, 중간값, 최대값)
- **이모티콘/이모지 사용 통계**: 반응 패턴 분석

### 💬 대화 역학 분석
- **대화 턴 분석**: 자동 대화 세션 구분
- **주제 전환 탐지**: 임베딩 기반 주제 변화 감지
- **참여자 패턴**: 대화 시작자, 응답자 분석 (시작 횟수 순으로 정렬)
- **상호작용 분석**: 사용자 간 소통 패턴
- **멘션 네트워크 분석**: @ 멘션을 통한 사용자 간 언급 관계 및 상호 멘션 분석

### 🎯 재미있는 지표
- **참여 불평등**: 지니 계수를 통한 발언권 분석
- **답장 속도**: 반응 시간 통계 및 분포
- **활동 연속성**: 연속 메시지 및 활동일 분석
- **심야 채팅**: 밤샘 대화 패턴 분석

### 🔍 고급 분석
- **임베딩 기반 주제 분석**: 의미론적 유사도를 통한 주제 세그먼트
- **SPLADE 하이브리드 검색**: Dense + Sparse 벡터를 활용한 대화 검색
- **시계열 분석**: 시간에 따른 대화 패턴 변화
- **감정 분석**: 대화 분위기와 감정 패턴 추적
- **스마트 토픽 클러스터링**: 고급 주제 분류 및 테마 분석
- **대화 리듬 분석**: 응답 패턴과 시간적 리듬 분석
- **대규모 데이터셋 최적화**: 지능형 청킹과 메모리 효율성

### 📈 시각화 (개선됨!)
- **활동 히트맵**: 시간대 × 요일 활동 분포
- **일별 타임라인**: 메시지 수 추이 그래프
- **주제 타임라인**: 주제 세그먼트 시각화 (겹침 문제 해결)
- **워드클라우드**: 한국어 폰트 지원으로 깨짐 현상 해결
- **사용자 활동 차트**: 많은 사용자도 깔끔하게 표시 (겹침 문제 해결)

## 🛠️ 설치 및 실행

### 📦 PyPI를 통한 간편 설치 (권장)

```bash
# 기본 설치
pip install kakao-analyzer

# 전체 기능 설치 (AI 모델 포함)
pip install kakao-analyzer[full]

# 개발 도구 포함 설치
pip install kakao-analyzer[dev]
```

설치 후 바로 사용 가능:
```bash
kakao-analyzer --input your_chat.csv
```

### 💻 개발자용 설치

개발이나 커스터마이징을 원한다면:

```bash
# 저장소 클론
git clone https://github.com/xistoh162108/kakaotalk_analyzer.git
cd kakaotalk_analyzer

# 의존성 설치
pip install -r requirements.txt

# 개발 모드로 설치
pip install -e .
```

### 📋 요구사항
- **Python**: 3.8 이상
- **운영체제**: Windows, macOS, Linux 모두 지원
- **메모리**: 최소 4GB RAM (대용량 데이터는 8GB+ 권장)

## 🚀 빠른 시작

### 1단계: 설치
```bash
pip install kakao-analyzer
```

### 2단계: 카카오톡 대화 내보내기
1. 카카오톡 앱에서 분석하고 싶은 대화방 열기
2. `☰` 메뉴 → `채팅방 설정` → `대화 내용 내보내기`
3. `텍스트 파일(.txt)` 선택하여 내보내기
4. 내보낸 파일을 CSV 형식으로 변환 (또는 직접 CSV로 준비)

### 3단계: 분석 실행
```bash
# 기본 분석
kakao-analyzer --input your_chat.csv

# 고급 AI 분석 포함 (권장)
kakao-analyzer --input your_chat.csv --use-splade --use-ollama
```

### ✨ 사용법
```bash
# 기본 분석 (가장 간단)
kakao-analyzer --input chat.csv

# 출력 디렉토리 지정
kakao-analyzer --input chat.csv --outdir my_analysis

# 도움말 보기
kakao-analyzer --help
```

### 🛠️ 고급 옵션
```bash
kakao-analyzer --input INPUT [옵션들]

필수 옵션:
  --input INPUT                입력 CSV 파일 경로

분석 옵션:
  --window-minutes MINUTES     대화 턴 구분 시간 (기본값: 30분)
  --topic-window-size SIZE     주제 분석 윈도우 크기 (기본값: 15)
  --similarity-threshold FLOAT 유사도 임계값 (기본값: 0.3)
  --batch-size SIZE            임베딩 배치 크기 (기본값: 32)
  --max-workers WORKERS        최대 병렬 작업 수 (기본값: 4)

AI 모델 옵션:
  --use-ollama                 Ollama 임베딩 사용 (로컬)
  --model-name MODEL           Ollama 모델명 (기본값: gpt-oss:20b)
  --embed-model MODEL          임베딩 모델명 (기본값: bge-m3)
  --use-splade                 SPLADE 희소 검색 활성화

대규모 데이터 옵션:
  --subset-size SIZE           테스트용 부분집합 크기  
  --subset-strategy STRATEGY   부분집합 전략 (balanced/temporal/diverse/random)

카카오톡 특화 옵션:
  --disable-message-grouping   연속 메시지 그룹핑 비활성화 (비추천)
  --group-window-seconds N     메시지 그룹핑 시간 창 (기본값: 60초)

출력 옵션:
  --outdir DIRECTORY           출력 디렉토리
  --figure-dpi DPI             그래프 해상도 (기본값: 300)
  --log-level LEVEL            로그 레벨 (DEBUG/INFO/WARNING/ERROR)
  --quiet                      콘솔 출력 억제
  --skip-viz                   시각화 생성 건너뛰기
```

### 💡 사용 예제
```bash
# 🌟 기본 분석 (추천)
kakao-analyzer --input chat.csv

# 🤖 AI 기능 전체 활용
kakao-analyzer --input chat.csv --use-splade --use-ollama --model-name "gpt-oss:20b"

# 📊 특정 디렉토리에 결과 저장
kakao-analyzer --input chat.csv --outdir my_results

# ⚡ 대용량 데이터 빠른 테스트
kakao-analyzer --input big_chat.csv --subset-size 1000 --subset-strategy balanced

# 🔧 세밀한 설정 조정
kakao-analyzer --input chat.csv --topic-window-size 20 --similarity-threshold 0.4

# 🎨 고해상도 그래프 생성
kakao-analyzer --input chat.csv --figure-dpi 600

# 🔇 조용한 실행 (CI/CD용)
kakao-analyzer --input chat.csv --quiet --skip-viz --log-level ERROR
```

## 📁 출력 구조

분석 완료 후 다음과 같은 구조로 결과가 생성됩니다:

```
output_directory/
├── summary/                    # 요약 정보
│   ├── analysis_summary.json   # 전체 분석 요약
│   └── processed_messages.csv  # 처리된 메시지 데이터
├── basics/                     # 기본 통계
│   └── basic_statistics.json   # 상세 통계 데이터
├── context/                    # 맥락 분석
│   ├── topic_segments.json     # 주제 세그먼트 데이터
│   ├── topic_segments.csv      # 주제 세그먼트 테이블
│   ├── mood_analysis.json      # 감정 및 분위기 분석
│   ├── advanced_topics.json    # 고급 주제 클러스터링
│   ├── rhythm_analysis.json    # 대화 리듬 분석
│   ├── mention_analysis.json   # 멘션 패턴 및 네트워크 분석
│   ├── hybrid_index.meta.json  # 검색 인덱스 (SPLADE)
│   └── splade_scores.parquet   # 희소 벡터 데이터 (SPLADE)
├── fun/                        # 재미 지표
│   └── fun_metrics.json        # 참여 불평등, 답장 속도 등
├── figures/                    # 시각화
│   ├── heatmap_hour_weekday.png
│   ├── timeseries_daily.png
│   ├── topic_timeline.png
│   ├── wordcloud_global.png
│   ├── user_activity_distribution.png
│   └── reply_latency_distribution.png
├── reports/                    # 리포트
│   ├── analysis_report.md      # 종합 분석 리포트
│   └── qa_test_report.md       # 품질 보증 리포트 (테스트시)
└── logs/                       # 로그
    ├── pipeline_metadata.json  # 실행 메타데이터
    └── analyzer.log            # 분석 로그
```

## 📋 입력 데이터 형식

카카오톡에서 내보낸 CSV 파일이 필요합니다. 지원하는 컬럼 형식:

### 필수 컬럼
- `user` 또는 `User` 또는 `사용자`: 메시지 작성자
- `message` 또는 `Message` 또는 `메시지`: 메시지 내용  
- `datetime` 또는 `Date` 또는 `시간`: 메시지 시간

### 예제 CSV 형식
```csv
user,message,datetime
김민수,안녕하세요!,2024-01-01 10:00:00
이지은,네 안녕하세요! ㅎㅎ,2024-01-01 10:01:00
박서준,오늘 날씨 좋네요,2024-01-01 10:02:00
```

## 🧪 테스트

### 기본 기능 테스트
```bash
# 기본 기능 테스트 (의존성 없이)
python simple_test.py

# 전체 테스트 스위트 (의존성 설치 후)
python test_runner.py
```

### 테스트 데이터 생성
```bash
# 1000개 메시지 테스트 데이터 생성
python test_runner.py --generate-data
```

## 🔧 고급 설정

### 🚀 GPU 가속 지원 (NEW!)
- **자동 GPU 인식**: CUDA가 설치된 시스템에서 자동으로 GPU 사용
- **메모리 최적화**: GPU 메모리의 75%까지 자동 제한하여 안정성 보장
- **모델 자동 이관**: 임베딩 및 SPLADE 모델을 GPU로 자동 로드
- **배치 크기 최적화**: GPU 메모리에 따라 배치 크기 자동 조정

#### GPU 메모리별 최적화
- **12GB+ (RTX 3080Ti/4080 이상)**: 배치 크기 128, Half precision
- **8-12GB (RTX 3070/4070)**: 배치 크기 64, Half precision  
- **4-8GB (GTX 1660/RTX 3060)**: 배치 크기 48, Mixed precision
- **4GB 미만**: CPU 모드로 자동 전환

### 임베딩 모델
- **기본값**: Mock 임베딩 (외부 의존성 없음)
- **고급 옵션**: SentenceTransformer 모델 (`sentence-transformers` 설치 필요)
- **GPU 가속**: CUDA 지원 시 자동으로 GPU에서 실행

### SPLADE 하이브리드 검색
- **활성화**: `--use-splade` 옵션 사용
- **기능**: Dense + Sparse 벡터를 결합한 고급 검색
- **용도**: 특정 주제나 키워드로 대화 검색
- **GPU 가속**: Half precision으로 GPU에서 고속 처리

### Ollama 연동 (한국어 모델 완전 지원)
- **활성화**: `--use-ollama` 옵션 사용
- **요구사항**: Ollama 서버 실행 중이어야 함
- **기본 모델**: gpt-oss:20b (추천)

#### 🇰🇷 지원 한국어 모델들
| 모델명 | 크기 | 특징 | 사용법 |
|--------|------|------|--------|
| `gpt-oss:20b` | 20B | 범용 한국어 모델 (기본값) | `--model-name "gpt-oss:20b"` |
| `gpt-oss:40b` | 40B | 고성능 한국어 모델 | `--model-name "gpt-oss:40b"` |
| `llama3-ko` | 8B | LLaMA3 한국어 파인튜닝 | `--model-name "llama3-ko"` |
| `aya:8b` | 8B | 다국어 지원 (한국어 포함) | `--model-name "aya:8b"` |
| `gemma2-ko` | 9B | Google Gemma2 한국어 버전 | `--model-name "gemma2-ko"` |
| `solar-ko` | 10.7B | Upstage SOLAR 한국어 모델 | `--model-name "solar-ko"` |
| `eeve-korean` | 10.8B | 한국 전용 대화 모델 | `--model-name "eeve-korean"` |
| `beomi-ko` | 13B | 한국어 특화 고성능 모델 | `--model-name "beomi-ko"` |

#### 🚀 모델 설치 및 사용법
```bash
# Ollama 설치 (한 번만)
curl -fsSL https://ollama.ai/install.sh | sh

# 한국어 모델 다운로드 (예: gpt-oss:20b)
ollama pull gpt-oss:20b

# 분석 실행
kakao-analyzer --input chat.csv --use-ollama --model-name "gpt-oss:20b"
```

## 📊 분석 결과 해석

### 참여 불평등 (지니 계수)
- **0.0-0.3**: 매우 균등한 참여
- **0.3-0.5**: 약간 불균등
- **0.5-0.7**: 불균등한 참여
- **0.7-1.0**: 매우 불균등 (소수가 대화 주도)

### 멘션 분석 지표
- **상호 멘션 관계**: 서로 멘션하는 사용자 쌍의 수와 빈도
- **멘션 네트워크 밀도**: 전체 가능한 멘션 관계 대비 실제 멘션 관계 비율
- **멘션 목적 분류**: 질문, 부탁/요청, 인사, 긴급, 토론/의견 등으로 자동 분류

### 답장 속도 카테고리
- **즉석 답장**: 1분 이내
- **빠른 답장**: 1-5분
- **보통 답장**: 5-30분
- **느린 답장**: 30분 이상

### 주제 세그먼트
- 임베딩 유사도 기반으로 자동 감지
- 각 세그먼트별 키워드, 참여자, 지속시간 제공
- 주제 전환 패턴 분석

## 🎨 시각화 가이드

모든 그래프는 한국어 폰트를 지원하며, 다음 형식으로 저장됩니다:
- **형식**: PNG (기본값)
- **해상도**: 300 DPI (기본값)
- **크기**: 자동 조정

### 주요 시각화
1. **활동 히트맵**: 언제 가장 활발한지 한눈에 파악
2. **일별 타임라인**: 시간에 따른 대화량 변화
3. **주제 타임라인**: 대화 주제의 시간적 흐름
4. **워드클라우드**: 자주 사용된 단어들
5. **사용자 활동**: 각 참여자의 기여도
6. **답장 속도**: 소통 스타일 분석

## 💡 활용 사례

### 개인/그룹 분석
- 친구들과의 대화 패턴 분석
- 가족 단톡방 활동 분석
- 동호회/커뮤니티 소통 패턴

### 연구 목적
- 디지털 커뮤니케이션 연구
- 사회 네트워크 분석
- 언어학 연구 데이터

### 비즈니스 분석
- 팀 커뮤니케이션 효율성
- 고객 소통 패턴
- 커뮤니티 관리 인사이트

## 🛡️ 개인정보 보호

- 모든 분석은 로컬에서 수행됩니다
- 데이터는 외부로 전송되지 않습니다
- 결과 파일에는 원본 메시지가 포함될 수 있으니 공유 시 주의하세요
- 개인 식별 정보가 포함된 분석 결과는 신중히 관리하세요

## 🐛 문제 해결

### 일반적인 문제

**1. 모듈을 찾을 수 없음**
```bash
pip install -r requirements.txt
```

**2. 한국어 폰트 문제**
- macOS: Apple Gothic 자동 사용
- Windows: Malgun Gothic 자동 사용
- Linux: Noto Sans CJK KR 설치 권장

**3. 메모리 부족**
- 큰 데이터셋의 경우 `--topic-window-size` 값을 줄여보세요
- 임베딩 비활성화: 설정에서 `use_embeddings = False`

**4. CSV 인코딩 문제**
- 자동 인코딩 감지 기능 내장
- 수동 지정이 필요한 경우 파일을 UTF-8로 변환

### 로그 확인
```bash
# 상세 로그로 실행
kakao-analyzer --input chat.csv --log-level DEBUG

# 로그 파일 확인 (기본 출력 디렉토리)
cat chat_analysis/logs/run.log
```

## 📈 성능 최적화

### 대용량 데이터 처리
- **배치 크기 조정**: 설정에서 `batch_size` 변경
- **임베딩 비활성화**: 기본 분석만 필요한 경우
- **윈도우 크기 감소**: 메모리 사용량 절약

### 속도 향상
- **병렬 처리**: 가능한 곳에서 자동 적용
- **캐싱**: 중간 결과 자동 저장
- **최적화된 알고리즘**: 대용량 데이터 대응

## 🚧 로드맵

현재 이 프로젝트는 지속적으로 개발 중이며, 다음과 같은 기능들이 추가될 예정입니다:

### 🔮 예정된 기능들
- **🌐 다중 메신저 지원**: 라인, 텔레그램, 디스코드 등 다른 메신저 플랫폼 지원
- **📱 웹 인터페이스**: 브라우저에서 사용할 수 있는 직관적인 UI
- **⚡ 실시간 분석**: 대화 진행 중 실시간 패턴 분석
- **🤖 고급 AI 기능**: GPT-4 통합, 더 정교한 감정 분석
- **📊 대시보드**: 실시간 모니터링 및 알림 시스템
- **🔍 검색 엔진**: 대화 내용 고급 검색 및 필터링
- **📈 트렌드 분석**: 시간에 따른 대화 패턴 변화 추적
- **🎨 테마 커스터마이제이션**: 시각화 스타일 및 색상 테마 선택
- **💾 클라우드 백업**: 분석 결과 자동 백업 및 동기화
- **📧 리포트 자동 발송**: 정기적인 분석 리포트 이메일 발송

### 🎯 다음 버전 (v1.1.0) 예정 기능
- 음성 메시지 길이 분석
- 파일 공유 패턴 분석
- 대화방 참여자 변화 추적
- 감정 변화 시계열 분석
- 키워드 알림 시스템

## 👨‍💻 개발자

### 🎓 주 개발자

- **xistoh162108** (KAIST)
  - 📧 이메일: xistoh162108@kaist.ac.kr
  - 🏫 소속: 한국과학기술원 (KAIST)
  - 🔗 GitHub: [@xistoh162108](https://github.com/xistoh162108)

### 🤖 AI 협력 개발

- **Claude AI** (Anthropic)
  - 코드 아키텍처 설계 및 구현
  - 알고리즘 최적화 및 한국어 특화 기능
  - 문서화 및 사용자 경험 개선

이 프로젝트는 **인간-AI 협업**의 결과물로, 학술 연구와 실용적 도구 개발의 만남을 보여줍니다.

## 🤝 기여하기

### 🔗 협업 방식

- **이슈 리포팅**: [GitHub Issues](https://github.com/xistoh162108/kakaotalk_analyzer/issues)
- **기능 제안**: Pull Request 환영
- **학술 협력**: 연구 목적 사용 시 인용 부탁드립니다

### 💡 개선 아이디어

- 새로운 분석 지표 추가
- 다른 메신저 지원 (라인, 텔레그램 등)
- 웹 인터페이스 개발
- 실시간 분석 기능

### 📝 인용 (Citation)

학술 논문이나 연구에서 사용하실 때:

```bibtex
xistoh162108, & Claude AI. (2025). Kakao Talk Analyzer: 
Comprehensive analysis tool for KakaoTalk chat exports (v1.0.1). 
https://github.com/xistoh162108/kakaotalk_analyzer
```

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.

## 🙏 감사의 말

- **한국어 자연어 처리**: KoNLPy, mecab-python3
- **시각화**: matplotlib, seaborn, wordcloud
- **데이터 처리**: pandas, numpy
- **머신러닝**: scikit-learn, sentence-transformers

---

**🤖 Generated with [Claude Code](https://claude.ai/code)**

Co-Authored-By: Claude <noreply@anthropic.com>
