"""The web UI is served by the same Flask app, so the API must not be open to other websites."""
import flask
import pytest

import server

EVIL = {"Origin": "https://evil.example", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type"}


def test_another_website_gets_no_cross_origin_permission():
    """The browser asks first (a preflight) before a page on another site may POST JSON to this
    server; without an Access-Control-Allow-Origin answer it never sends the real request."""
    client = server.app.test_client()
    for path in ("/api/config", "/api/generate/youtube", "/api/usage/reset"):
        r = client.options(path, headers=EVIL)
        assert "Access-Control-Allow-Origin" not in r.headers, path


def test_a_normal_response_carries_no_cors_headers_either():
    r = server.app.test_client().get("/api/config", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in r.headers


def test_the_ui_itself_is_still_served():
    assert server.app.test_client().get("/").status_code == 200


def test_cors_can_be_opted_into_for_named_origins_only():
    app = flask.Flask("t")

    @app.route("/x", methods=["GET", "POST"])
    def x():
        return "ok"

    server.enable_cors(app, "http://localhost:3000, https://my.tool")
    client = app.test_client()
    allowed = client.get("/x", headers={"Origin": "http://localhost:3000"})
    assert allowed.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
    refused = client.get("/x", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in refused.headers


@pytest.mark.parametrize("value", ["", "  ", None, " , "])
def test_nothing_is_enabled_without_origins(value):
    app = flask.Flask("t")
    server.enable_cors(app, value)
    assert "flask_cors" not in str(app.after_request_funcs)
