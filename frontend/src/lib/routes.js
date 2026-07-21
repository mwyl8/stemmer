// The frontend's client-side job-result route deliberately does NOT live at
// /jobs/:jobId, even though that would read naturally next to the backend's
// /jobs/{id} API. vite.config.js's dev proxy (and any real deployment
// fronting this app the same way -- API prefix -> backend, everything else
// -> the SPA) forwards by path *prefix*, and can't distinguish "a browser
// navigating to the SPA route" from "an XHR hitting the API" -- both are
// just `GET /jobs/<id>`. A colliding path only breaks on a fresh/full page
// load (a shared link, a new tab, a private window) -- client-side
// react-router navigation never round-trips through the proxy, so the bug
// stayed invisible until someone actually opened a copied link. Keeping the
// frontend route on its own prefix removes the ambiguity structurally
// instead of trying to sniff Accept headers in the proxy.
export function jobRoute(jobId) {
  return `/results/${jobId}`
}
