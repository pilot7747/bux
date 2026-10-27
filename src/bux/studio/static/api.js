// The studio's JSON API. The session token comes from the page URL (?t=...) and
// goes with every call, so that no other page can read or write the file.

const token = new URLSearchParams(location.search).get("t") ?? "";

/** A reply that is not 2xx: its status, and its JSON body if it has one. */
export class ApiError extends Error {
  constructor(status, body) {
    super(body?.error ?? `HTTP ${status}`);
    this.status = status;
    this.body = body;
  }
}

/** Call the API: POST `body` as JSON, or GET without one. Returns the JSON reply. */
export async function api(path, body) {
  const headers = { "X-Bux-Token": token };
  const init = { method: body === undefined ? "GET" : "POST", headers };
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const res = await fetch(path, init);
  const json = res.headers.get("Content-Type")?.includes("json") ? await res.json() : null;
  if (!res.ok) throw new ApiError(res.status, json);
  return json;
}
