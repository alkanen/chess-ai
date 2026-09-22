import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chess_ai.config import Config, ServerConfig
from chess_ai.web import create_app
from chess_ai.web.app import STATIC_DIR

FRONTEND_FIXTURES = Path(__file__).parents[1] / "frontend" / "src" / "test" / "fixtures"

INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <script type="module" crossorigin src="./assets/index.js"></script>
  </head>
  <body><div id="root"></div></body>
</html>
"""


@pytest.fixture
def static_dir(tmp_path):
    """A stand-in for the built frontend."""
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (tmp_path / "assets" / "index.js").write_text("console.log('board');", encoding="utf-8")
    (tmp_path / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    return tmp_path


def client(prefix: str, static_dir) -> TestClient:
    config = Config(server=ServerConfig(path_prefix=prefix))
    return TestClient(create_app(config, static_dir=static_dir))


@pytest.fixture
def chess_client(static_dir):
    return client("/chess", static_dir)


def test_start_position_under_prefix(chess_client):
    response = chess_client.get("/chess/api/start-position")

    assert response.status_code == 200
    view = response.json()
    assert view["turn"] == "white"
    assert len(view["pieces"]) == 32
    assert view["pieces"]["e1"] == {"color": "white", "type": "king"}
    assert view["pieces"]["e8"] == {"color": "black", "type": "king"}


def test_frontend_fixture_matches_start_position(chess_client):
    """The Vitest fixture must stay what the server actually sends."""
    fixture = json.loads((FRONTEND_FIXTURES / "start-position.json").read_text(encoding="utf-8"))

    assert chess_client.get("/chess/api/start-position").json() == fixture


def test_page_under_prefix_gets_base_href(chess_client):
    response = chess_client.get("/chess/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<head><base href="/chess/">' in response.text
    assert 'src="./assets/index.js"' in response.text


def test_static_files_under_prefix(chess_client):
    assert chess_client.get("/chess/assets/index.js").text == "console.log('board');"
    assert chess_client.get("/chess/favicon.svg").status_code == 200


def test_prefix_without_trailing_slash_redirects(chess_client):
    response = chess_client.get("/chess", follow_redirects=False)

    assert response.status_code in (301, 302, 307, 308)
    assert response.headers["location"] == "/chess/"


@pytest.mark.parametrize(
    "path",
    ["/", "/api/start-position", "/assets/index.js", "/favicon.svg", "/openapi.json", "/docs"],
)
def test_nothing_is_served_outside_prefix(chess_client, path):
    assert chess_client.get(path, follow_redirects=False).status_code == 404


def test_api_docs_under_prefix(chess_client):
    assert chess_client.get("/chess/api/openapi.json").status_code == 200
    assert chess_client.get("/chess/api/docs").status_code == 200


def test_nested_prefix(static_dir):
    apps_client = client("/apps/chess", static_dir)

    assert apps_client.get("/apps/chess/api/start-position").status_code == 200
    assert '<base href="/apps/chess/">' in apps_client.get("/apps/chess/").text


def test_empty_prefix_serves_at_root(static_dir):
    root_client = client("", static_dir)

    assert root_client.get("/api/start-position").json()["turn"] == "white"
    assert '<base href="/">' in root_client.get("/").text
    assert root_client.get("/assets/index.js").status_code == 200


def test_unbuilt_frontend_is_explained_but_api_works(tmp_path):
    unbuilt_client = client("/chess", tmp_path / "missing")

    response = unbuilt_client.get("/chess/")

    assert response.status_code == 503
    assert "build-frontend" in response.text
    assert unbuilt_client.get("/chess/api/start-position").status_code == 200


@pytest.mark.skipif(not (STATIC_DIR / "index.html").is_file(), reason="frontend not built")
def test_built_frontend_uses_relative_urls():
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    urls = re.findall(r'(?:src|href)="([^"]*)"', page)

    assert urls
    assert all(url.startswith("./") for url in urls), urls
