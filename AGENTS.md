# AGENTS.md

## Python 환경 — uv 기반 가상환경 사용

이 프로젝트는 **uv**로 의존성과 가상환경을 관리한다. 시스템 `python` / `python3`를
직접 호출하지 말고, 항상 uv를 통해 실행한다.

- 명령 실행: `uv run python ...` (예: `uv run python generate_data.py`)
- 동기화/설치: `uv sync` (의존성 동기화), `uv add <pkg>` (런타임 의존성 추가),
  `uv add --dev <pkg>` (개발 의존성 추가)
- 가상환경은 `.venv/`에 있으며 uv가 관리한다. `pip install`을 직접 쓰지 않는다.

### 테스트 실행

테스트는 `unittest` 기반이다 (`tests/` 디렉터리).

```bash
uv run python -m unittest discover -s tests -v      # 전체
uv run python -m unittest tests.test_local_planner -v   # 단일 모듈
```

`pytest`는 현재 의존성에 포함되어 있지 않다. pytest로 돌리려면 먼저
`uv add --dev pytest` 후 `uv run pytest tests/ -v`.
