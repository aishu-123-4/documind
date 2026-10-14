# HTTP Basics

HTTP (HyperText Transfer Protocol) is the application-layer protocol that powers
the web. A client sends a request to a server, and the server returns a
response. HTTP is stateless: each request is independent unless state is added
with cookies or tokens.

## Methods

The most common HTTP methods are GET (retrieve a resource), POST (submit data
and create a resource), PUT (replace a resource), PATCH (partially update), and
DELETE (remove a resource). GET requests should be safe and idempotent — they
must not change server state.

## Status codes

Responses carry a status code. 2xx means success (200 OK, 201 Created). 3xx
means redirection (301 Moved Permanently, 304 Not Modified). 4xx means a client
error (400 Bad Request, 401 Unauthorized, 404 Not Found). 5xx means a server
error (500 Internal Server Error, 503 Service Unavailable).

## Headers

Headers carry metadata. Content-Type describes the body's media type,
Authorization carries credentials, and Cache-Control governs caching behavior.
