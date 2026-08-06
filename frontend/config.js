// RAG Document Search — API base URL.
//
// LOCAL DEV (frontend and backend on the same host): leave this as "".
//   The frontend will call the backend on the same origin.
//
// SEPARATE HOSTING (frontend static host + backend API host):
//   Set this to the deployed backend URL WITHOUT a trailing slash, e.g.:
//     "https://my-backend.onrender.com"
window.API_BASE = "https://rag-doc-search-1.onrender.com";