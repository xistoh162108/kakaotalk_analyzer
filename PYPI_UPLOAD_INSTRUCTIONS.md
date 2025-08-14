# PyPI 업로드 가이드

## 패키지 준비 완료 ✅

kakao-analyzer 패키지가 성공적으로 빌드되고 테스트되었습니다!

- 📦 **Source Distribution**: `dist/kakao_analyzer-1.0.0.tar.gz`
- 🔧 **Wheel Package**: `dist/kakao_analyzer-1.0.0-py3-none-any.whl`
- ✅ **패키지 검증**: 통과
- ✅ **로컬 설치 테스트**: 통과
- ✅ **CLI 스크립트**: `kakao-analyzer` 명령어 정상 작동
- 📄 **MIT 라이선스**: LICENSE 파일 포함
- 📖 **업데이트된 README**: 로드맵 및 예정 기능 추가

## PyPI 업로드 방법

### 1. PyPI 계정 준비

1. [PyPI 웹사이트](https://pypi.org/)에서 계정을 생성하세요
2. [API 토큰](https://pypi.org/manage/account/token/)을 생성하세요
3. (선택사항) [TestPyPI](https://test.pypi.org/)에서 먼저 테스트해보세요

### 2. 업로드 명령어

#### TestPyPI 업로드 (테스트)
```bash
python -m twine upload --repository testpypi dist/*
```

#### 실제 PyPI 업로드
```bash
python -m twine upload dist/*
```

### 3. API 토큰 사용

업로드 시 인증 정보를 묻는다면:
- **Username**: `__token__`
- **Password**: 생성한 API 토큰 (pypi-로 시작)

### 4. 설치 확인

업로드 후 설치 테스트:

```bash
# TestPyPI에서 설치
pip install --index-url https://test.pypi.org/simple/ kakao-analyzer

# 실제 PyPI에서 설치
pip install kakao-analyzer
```

## 사용법

패키지 설치 후:

```bash
# 기본 분석
kakao-analyzer --input your_chat_file.csv

# 고급 AI 분석 포함
kakao-analyzer --input your_chat_file.csv --use-splade --use-ollama

# 옵션 확인
kakao-analyzer --help
```

## 패키지 특징

### 🚀 완전 자동화된 카카오톡 대화 분석
- **기본 통계**: 사용자별 활동, 시간대별 패턴, 메시지 특성
- **고급 분석**: SPLADE 희소 검색, Ollama LLM 통합, 토픽 감정 분석
- **멘션 분석**: @ 멘션 네트워크, 응답 패턴, 대화 맥락
- **시각화**: 히트맵, 타임라인, 워드클라우드, 네트워크 그래프
- **한국어 최적화**: 카카오톡 특성 고려 (메시지 그룹화, 시스템 메시지 필터링)

### 🛠️ 기술 스택
- **데이터 처리**: pandas, numpy
- **ML/AI**: scikit-learn, sentence-transformers, transformers, torch
- **시각화**: matplotlib, seaborn, wordcloud, networkx
- **한국어 지원**: 완전한 한국어 인터페이스

### 📈 성능 최적화
- **대용량 데이터셋**: 지능적 샘플링 및 청킹
- **메모리 효율성**: 배치 처리 및 스트리밍
- **병렬 처리**: 멀티스레딩 지원

## 문제 해결

### 종속성 이슈
필요한 경우 추가 종속성을 설치하세요:

```bash
# 전체 기능 사용
pip install "kakao-analyzer[full]"

# 개발 도구
pip install "kakao-analyzer[dev]"
```

### 한국어 폰트 이슈
macOS에서 시각화 시 폰트 문제가 있다면:

```bash
# 시스템 한글 폰트 확인
fc-list : family | grep -i korean
```

---

**🎉 패키지 준비 완료!** 

위 가이드에 따라 PyPI에 업로드하면 `pip install kakao-analyzer`로 전 세계 어디서나 설치 가능합니다.